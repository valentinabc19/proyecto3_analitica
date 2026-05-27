import math

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class S2Normalizer(nn.Module):
    """
    Normalización z-score por banda para tiles de Sentinel-2.
    Stats calculados sobre el dataset de Cali.
    """
    def __init__(self):
        super().__init__()
        mean = [1931.68, 3341.38, 4165.15, 1960.77]
        std  = [1462.26, 1425.28, 2059.46, 1589.12]
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 4, 1, 1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1, 4, 1, 1))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / (self.std + 1e-6)
 
 
class RemoteCLIP_Visual(nn.Module):
    """
    ViT-B/32 (RemoteCLIP/OpenAI) para Sentinel-2.
 
    INPUT : (B, 4, 64, 64)  — 4 bandas S2 en valores crudos
    OUTPUT: (B, 50, 768)    — [CLS + 49 patches] × 768 dims
                               (ViT-B/32: patches 32×32 → grilla 7×7 = 49)
 
    Pipeline:
        x → S2Normalizer (4 bandas) → band_projection (4→3) →
        interpolate (64→224) → ViT-B/32 sin pooling → (B, 50, 768)
    """
    def __init__(self):
        super().__init__()
        # Normalización z-score
        self.s2_norm = S2Normalizer()
 
        # Proyección aprendible 4→3 bandas (mezcla lineal aprendida)
        # El ViT espera 3 canales; en lugar de descartar una banda,
        # aprendemos qué combinación es más informativa para contaminación
        self.band_projection = nn.Conv2d(4, 3, kernel_size=1, bias=False)
 
        # Cargar Backbone ViT-B/32 preentrenado
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.clip_vision = model.visual
        
        # ── NUEVA LÓGICA DE FINE-TUNING PARCIAL (A PRUEBA DE SOBREAJUSTE) ──
        # 1. Congelar inicialmente todo el codificador visual de RemoteCLIP
        for param in self.clip_vision.parameters():
            param.requires_grad = False
            
        # 2. Descongelar únicamente los últimos 2 bloques residuales del Transformer (Bloques 10 y 11)
        num_blocks = len(self.clip_vision.transformer.resblocks)
        for i in range(num_blocks - 2, num_blocks):
            for param in self.clip_vision.transformer.resblocks[i].parameters():
                param.requires_grad = True
                
        # 3. Descongelar el LayerNorm final posterior al transformer
        for param in self.clip_vision.ln_post.parameters():
            param.requires_grad = True
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Normalizar las 4 bandas
        x = self.s2_norm(x)  
 
        # 2. Proyectar 4 → 3 canales (aprendible)
        x = self.band_projection(x)         
 
        # 3. Interpolar al tamaño que espera el ViT-B/32
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
 
        # 4. Forward del ViT sin pooling final (extraemos tokens)
        x = self.clip_vision.conv1(x)                          # (B, 768, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1)             # (B, 768, 49)
        x = x.permute(0, 2, 1)                                 # (B, 49, 768)
 
        cls = (self.clip_vision.class_embedding.to(x.dtype)
               + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device))
        x = torch.cat([cls, x], dim=1)                  
        x = x + self.clip_vision.positional_embedding.to(x.dtype)
        x = self.clip_vision.ln_pre(x)
 
        x = x.permute(1, 0, 2)                              
        x = self.clip_vision.transformer(x)
        x = x.permute(1, 0, 2)                                # (B, 50, 768)
 
        return x 
 
 
class S5PMLP(nn.Module):
    """
    MLP encoder para Sentinel-5P optimizado para 6 canales (Gas + Máscara).
    """
    def __init__(self, d_hidden: int = 128, d_out: int = 256):
        super().__init__()
        # Registramos buffers de normalización para los 3 gases (índices 0, 2 y 4)
        self.register_buffer("mean", torch.tensor([2.5e-4, 1.8e-5, 0.13], dtype=torch.float32))
        self.register_buffer("std",  torch.tensor([1.2e-4, 9.0e-6, 0.02], dtype=torch.float32))
 
        # Cada MLP recibe 2 entradas: (valor_gas, mascara_confianza)
        self.mlp_no2 = self._build_mlp(d_hidden, d_out)
        self.mlp_so2 = self._build_mlp(d_hidden, d_out)
        self.mlp_o3  = self._build_mlp(d_hidden, d_out)
 
        self.pos_embedding = nn.Parameter(torch.randn(3, d_out) * 0.02)
        self.norm = nn.LayerNorm(d_out)
 
    @staticmethod
    def _build_mlp(d_hidden: int, d_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden * 2, d_out),
        )
 
    def forward(self, s5p_raw: torch.Tensor) -> torch.Tensor:
        # s5p_raw: (B, 6)
        
        # Normalizar los gases (columnas 0, 2, 4)
        no2_norm = (s5p_raw[:, 0] - self.mean[0]) / (self.std[0] + 1e-8)
        so2_norm = (s5p_raw[:, 2] - self.mean[1]) / (self.std[1] + 1e-8)
        o3_norm  = (s5p_raw[:, 4] - self.mean[2]) / (self.std[2] + 1e-8)
        
        # Concatenar con sus respectivas máscaras (columnas 1, 3, 5)
        feat_no2 = torch.stack([no2_norm, s5p_raw[:, 1]], dim=-1)
        feat_so2 = torch.stack([so2_norm, s5p_raw[:, 3]], dim=-1)
        feat_o3  = torch.stack([o3_norm,  s5p_raw[:, 5]], dim=-1)
 
        t_no2 = self.mlp_no2(feat_no2)    # (B, d_out)
        t_so2 = self.mlp_so2(feat_so2)    # (B, d_out)
        t_o3  = self.mlp_o3 (feat_o3)     # (B, d_out)
 
        tokens = torch.stack([t_no2, t_so2, t_o3], dim=1)     # (B, 3, 256)
        tokens = tokens + self.pos_embedding.unsqueeze(0)       
        return self.norm(tokens)
 
 
class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention: tokens S2 (Query) ← tokens S5P (Key/Value)
    Diagrama: Cross-Attention (S2—S5P) → tokens visuales atienden
              tokens espectrales → e_img_raw (512)
 
    Query  : tokens ViT   (B, 50, 768) → proyectados a (B, 50, 256)
    Key    : tokens S5P   (B, 3,  256)
    Value  : tokens S5P   (B, 3,  256)
 
    Salida : e_img_raw (B, 512) + attn_weights (B, 50, 3)
    """
    def __init__(self, d_vit: int = 768, d_s5p: int = 256, d_raw: int = 512):
        super().__init__()
 
        # Proyección Q: 768 → 256 (mismo espacio que K/V)
        self.proj_vit = nn.Linear(d_vit, d_s5p)
 
        # Cross-Attention (embed_dim = d_s5p = 256, 4 cabezas → 64 dims/cabeza)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_s5p, num_heads=4,
            dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(d_s5p)
 
        self.ffn = nn.Sequential(
            nn.Linear(d_s5p, d_s5p * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_s5p * 4, d_s5p),
        )
        self.norm2 = nn.LayerNorm(d_s5p)
 
        # Proyección del token CLS fusionado → e_img_raw (512)
        self.to_e_raw = nn.Sequential(
            nn.Linear(d_s5p, d_raw),
            nn.GELU(),
            nn.LayerNorm(d_raw),
        )
 
    def forward(self, vit_tokens: torch.Tensor, s5p_tokens: torch.Tensor):
        """
        vit_tokens : (B, 50, 768)
        s5p_tokens : (B, 3,  256)
        →  e_img_raw   (B, 512)
           attn_weights (B, 50, 3)  ← interpretabilidad
        """
        # 1. Proyectar Query al espacio de 256 dims
        q = self.proj_vit(vit_tokens)             # (B, 50, 256)
 
        # 2. Cross-Attention: Q=visual, K=V=espectral
        attn_out, attn_weights = self.cross_attn(
            query=q, key=s5p_tokens, value=s5p_tokens,
            need_weights=True, average_attn_weights=True
        )
        # attn_out     : (B, 50, 256)
        # attn_weights : (B, 50, 3)  → cuánto atiende cada patch a cada gas
 
        # 3. Residual + LayerNorm
        x = self.norm(q + attn_out)               # (B, 50, 256)
 
        # 4. FFN + residual
        x = self.norm2(x + self.ffn(x))           # (B, 50, 256)
 
        # 5. Token CLS (posición 0) → resumen global del tile fusionado
        cls_fused = x[:, 0, :]                    # (B, 256)
 
        # 6. Proyectar al espacio e_img_raw de 512 dims
        e_img_raw = self.to_e_raw(cls_fused)      # (B, 512)
 
        return e_img_raw, attn_weights

class SAEVisual(nn.Module):
    """
    Sparse AutoEncoder para la rama visual.
    Entrada: e_img_raw (B, 512) — salida del CrossAttentionFusion
    
    Encoder : e_img_raw → z_img (B, 512) sparse   [interpretabilidad]
    Decoder : z_img     → x_hat (B, 512)           [reconstrucción]
    Proyect.: e_img_raw → e_img (B, 256) L2-norm   [InfoNCE]
    
    Las neuronas de z_img aprenden a detectar conceptos visuales:
    zonas industriales, vegetación, densidad urbana, etc.
    """
    def __init__(self, d_raw: int = 512, d_out: int = 256, lambda_l1: float = 1e-3):
        super().__init__()
        self.lambda_l1 = lambda_l1

        self.encoder = nn.Sequential(
            nn.Linear(d_raw, d_raw),
            nn.ReLU()
        )
        self.decoder    = nn.Linear(d_raw, d_raw)
        self.projection = nn.Sequential(
            nn.Linear(d_raw, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, e_img_raw: torch.Tensor):
        """
        e_img_raw : (B, 512) — fusión imagen+gas del CrossAttention
        Retorna:
            z_img : (B, 512) sparse       → interpretabilidad visual
            x_hat : (B, 512)              → L_recon
            e_img : (B, 256) normalizado  → InfoNCE
        """
        z_img = self.encoder(e_img_raw)                           # (B, 512)
        x_hat = self.decoder(z_img)                               # (B, 512)
        e_img = F.normalize(self.projection(e_img_raw), dim=-1)   # (B, 256)
        return z_img, x_hat, e_img

    def sparsity_ratio(self, z_img: torch.Tensor, threshold: float = 0.01) -> float:
        """KPI: ≥ 0.70. Si es < 0.70 → aumentar lambda_l1."""
        return (z_img.abs() < threshold).float().mean().item()

    def neuron_analysis(self, z_by_class: dict, top_k: int = 10) -> dict:
        """
        Neuronas más activas por clase visual.
        
        Ejemplo de uso en el reporte:
            z_by_class = {
                "contaminacion_alta_NO2": z_industrial,  # (N, 512)
                "vegetacion_densa":       z_verde,        # (N, 512)
            }
            neuronas = sae_visual.neuron_analysis(z_by_class)
            # → {"contaminacion_alta_NO2": [12, 47, 203, ...],
            #    "vegetacion_densa":       [8,  91, 334, ...]}
            # Las neuronas [12, 47, 203] se activan en zonas industriales
            # pero no en zonas verdes → el SAE aprendió a detectar contaminación
        """
        return {
            clase: z.mean(dim=0).topk(top_k).indices.tolist()
            for clase, z in z_by_class.items()
        }
    
class TextEncoder(nn.Module):
    """
    XLM-RoBERTa con ajuste fino parcial de su última capa.
    """
    def __init__(self, model_name: str = 'xlm-roberta-base', d_raw: int = 512):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.proj = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, d_raw),
            nn.GELU(),
            nn.LayerNorm(d_raw),
        )
        
        # ── FINE-TUNING SELECCIONADO PARA TEXTO ────────────
        # 1. Congelar todo el backend de RoBERTa por defecto
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # 2. Descongelar únicamente la última capa del transformer (Capa 11)
        # XLM-RoBERTa-base tiene exactamente 12 capas de transformer (índices 0 a 11)
        num_layers = len(self.roberta.encoder.layer)
        for param in self.roberta.encoder.layer[num_layers - 1].parameters():
            param.requires_grad = True
            
        print(f"-> XLM-RoBERTa inicializado. Capa de transformer {num_layers-1} descongelada para Fine-Tuning.")
 
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (B, hidden_size)
        return self.proj(cls)                  # (B, 512)
    
class SAETexto(nn.Module):
    """
    Sparse AutoEncoder para la rama textual.
    Entrada: e_txt_raw (B, 512) — salida del TextEncoder (XLM-RoBERTa)

    Encoder : e_txt_raw → z_txt (B, 512) sparse   [interpretabilidad]
    Decoder : z_txt     → x_hat (B, 512)           [reconstrucción]
    Proyect.: e_txt_raw → e_txt (B, 256) L2-norm   [InfoNCE]

    Las neuronas de z_txt aprenden a detectar conceptos semánticos:
    términos de contaminación, ubicaciones geográficas, niveles de gas, etc.
    """
    def __init__(self, d_raw: int = 512, d_out: int = 256, lambda_l1: float = 1e-3):
        super().__init__()
        self.lambda_l1 = lambda_l1

        self.encoder = nn.Sequential(
            nn.Linear(d_raw, d_raw),
            nn.ReLU()
        )
        self.decoder    = nn.Linear(d_raw, d_raw)
        self.projection = nn.Sequential(
            nn.Linear(d_raw, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, e_txt_raw: torch.Tensor):
        """
        e_txt_raw : (B, 512) — salida proyectada de XLM-RoBERTa
        Retorna:
            z_txt : (B, 512) sparse       → interpretabilidad textual
            x_hat : (B, 512)              → L_recon
            e_txt : (B, 256) normalizado  → InfoNCE
        """
        z_txt = self.encoder(e_txt_raw)                           # (B, 512)
        x_hat = self.decoder(z_txt)                               # (B, 512)
        e_txt = F.normalize(self.projection(e_txt_raw), dim=-1)   # (B, 256)
        return z_txt, x_hat, e_txt

    def sparsity_ratio(self, z_txt: torch.Tensor, threshold: float = 0.01) -> float:
        return (z_txt.abs() < threshold).float().mean().item()

    def neuron_analysis(self, z_by_class: dict, top_k: int = 10) -> dict:
        """
        Neuronas más activas por clase textual.
        
        Comparar con SAEVisual.neuron_analysis() para encontrar
        neuronas cross-modal: conceptos que el modelo aprendió
        a representar igual en imagen y en texto.
        """
        return {
            clase: z.mean(dim=0).topk(top_k).indices.tolist()
            for clase, z in z_by_class.items()
        }
    
class GeoVisionCLIP(nn.Module):
    """
    Modelo completo — Situación 2 del proyecto.
 
    Rama visual:
        images → ViT → vit_tokens (B,50,768)
        s5p    → S5PMLP → s5p_tokens (B,3,256)
        CrossAttention(vit_tokens, s5p_tokens) → e_img_raw (B,512)
        SAEVisual(e_img_raw) → z_img, x_hat_img, e_img (B,256)
 
    Rama textual:
        texto → XLM-RoBERTa → e_txt_raw (B,512)
        SAETexto(e_txt_raw)  → z_txt, x_hat_txt, e_txt (B,256)
 
    Loss:
        L_InfoNCE(e_img, e_txt, τ)
        L_sae_img = MSE(x_hat_img, e_img_raw) + λ·||z_img||₁
        L_sae_txt = MSE(x_hat_txt, e_txt_raw) + λ·||z_txt||₁
        L_total   = L_InfoNCE + α·(L_sae_img + L_sae_txt)
    """
    def __init__(self, text_model_name: str = 'xlm-roberta-base'):
        super().__init__()
 
        # Rama visual
        self.vit         = RemoteCLIP_Visual()
        self.s5p_encoder = S5PMLP(d_hidden=128, d_out=256)
        self.fusion      = CrossAttentionFusion(d_vit=768, d_s5p=256, d_raw=512)
        self.sae_visual  = SAEVisual(d_raw=512, d_out=256, lambda_l1=1e-3)

        # Rama textual
        self.text_encoder = TextEncoder(model_name=text_model_name, d_raw=512)
        self.sae_texto    = SAETexto(d_raw=512, d_out=256, lambda_l1=2e-3)
        #                                                  ↑ puedes ajustar
        #                                                    independientemente
        #                                                    si uno no alcanza 0.70

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
 
    def forward(
        self,
        images:         torch.Tensor,   # (B, 4, 64, 64)
        s5p_raw:        torch.Tensor,   # (B, 3)
        input_ids:      torch.Tensor,   # (B, seq_len)
        attention_mask: torch.Tensor,   # (B, seq_len)
    ) -> dict:
 
        # ── Rama visual ───────────────────────────────────────────────
        vit_tokens  = self.vit(images)                        # (B, 50, 768)
        s5p_tokens  = self.s5p_encoder(s5p_raw)               # (B, 3,  256)
        e_img_raw, attn_weights = self.fusion(vit_tokens, s5p_tokens)  # (B, 512)
        z_img, x_hat_img, e_img = self.sae_visual(e_img_raw)  # (B,512),(B,512),(B,256)
 
        # ── Rama textual ──────────────────────────────────────────────
        e_txt_raw = self.text_encoder(input_ids, attention_mask)       # (B, 512)
        z_txt, x_hat_txt, e_txt = self.sae_texto(e_txt_raw)            # (B,512),(B,512),(B,256)
 
        return {
            # Embeddings para InfoNCE
            "e_img":        e_img,          # (B, 256) ← salida final visual
            "e_txt":        e_txt,          # (B, 256) ← salida final textual
            # Entradas al SAE (para L_recon)
            "e_img_raw":    e_img_raw,      # (B, 512)
            "e_txt_raw":    e_txt_raw,      # (B, 512)
            # Reconstrucciones (para L_recon)
            "x_hat_img":    x_hat_img,      # (B, 512)
            "x_hat_txt":    x_hat_txt,      # (B, 512)
            # Representaciones sparse (para interpretabilidad)
            "z_img":        z_img,          # (B, 512)
            "z_txt":        z_txt,          # (B, 512)
            # Temperatura y mapas de atención
            "logit_scale":  self.logit_scale.exp(),
            "attn_weights": attn_weights,   # (B, 50, 3) ← interpretabilidad por gas
        }