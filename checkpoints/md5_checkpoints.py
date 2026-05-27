import hashlib

checkpoint_path = "checkpoints\geovision_clip.pt"

def calcular_md5(filepath, chunk_size=8192):
    md5 = hashlib.md5()

    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()

md5_hash = calcular_md5(checkpoint_path)

# Guardar hash en archivo .md5
md5_file = checkpoint_path + ".md5"

with open(md5_file, "w") as f:
    f.write(md5_hash)

print(f"MD5: {md5_hash}")
print(f"Hash guardado en: {md5_file}")