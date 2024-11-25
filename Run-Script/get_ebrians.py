import os

from ebrains_drive import BucketApiClient

dataset_id = "d55146e8-fc86-44dd-95db-7191fdca7f30"
token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJLYU01NTRCM2RmMHBIamZYWi1aRl94bUUwMThPS1R0RkNjMjR3aVVqQmFvIn0.eyJleHAiOjE3MzI1NTg1MjUsImlhdCI6MTczMjUxNTMyNCwiYXV0aF90aW1lIjoxNzMyNTAzNjQ5LCJqdGkiOiIxM2NlZmVlNi1kYjBmLTRiMGItOGUzYy1jZWEyNGZjM2JjMDIiLCJpc3MiOiJodHRwczovL2lhbS5lYnJhaW5zLmV1L2F1dGgvcmVhbG1zL2hicCIsImF1ZCI6WyJqdXB5dGVyaHViLWpzYyIsInh3aWtpIiwidGVhbSIsImdyb3VwIl0sInN1YiI6IjJiYzcxNjIwLWI3NzktNGM3NS1hNGE4LWFiZTkxMTIxODBkNSIsInR5cCI6IkJlYXJlciIsImF6cCI6Imp1cHl0ZXJodWIiLCJzZXNzaW9uX3N0YXRlIjoiNjE4ZTJkYzgtYzMyOS00OTYwLTgyN2ItMTc2ZTJhM2FhZjJlIiwiYWxsb3dlZC1vcmlnaW5zIjpbImh0dHBzOi8vanVweXRlcmh1Yi5hcHBzLmpzYy5oYnAuZXUvIiwiaHR0cHM6Ly9sYWIuZWJyYWlucy5ldS8iLCJodHRwczovL2xhYi5qc2MuZWJyYWlucy5ldS8iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIl19LCJzY29wZSI6ImNvbGxhYi5kcml2ZSBwcm9maWxlIG9mZmxpbmVfYWNjZXNzIGNsYi53aWtpLndyaXRlIGVtYWlsIHJvbGVzIG9wZW5pZCBncm91cCBjbGIud2lraS5yZWFkIHRlYW0iLCJzaWQiOiI2MThlMmRjOC1jMzI5LTQ5NjAtODI3Yi0xNzZlMmEzYWFmMmUiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmFtZSI6Illvbmdkb25nIEZhbiIsIm1pdHJlaWQtc3ViIjoiMzIyNjAzMjgyODk3NjI3NCIsInByZWZlcnJlZF91c2VybmFtZSI6ImhpdGZ5ZCIsImdpdmVuX25hbWUiOiJZb25nZG9uZyIsImZhbWlseV9uYW1lIjoiRmFuIiwiZW1haWwiOiJoaXRmeWRAaGl0LmVkdS5jbiJ9.PotsiPtuzSG7aQ7K7xmP5UCha8p4yQWIEVcaAy6Q0rJbTm5SPvg17D9B2GXTosYZKoIEUxW4LTI5hLQaUedhTxmuUAFkwz2T1req5lEVoooE8TceLCRS5FTRv2FH40vbownD9M8euDQp2y22Vn9ziv7DhC3n3QuyJ0NhmbLlUNdXO7-tcvqaBLONkrgEX0xg147F1tsRyj5BfEeKFvoOtvKNX6lt-W7OeoFbZGH-hNvi-OtM6nd2gkNd-LCW4t0t1Ry3TlT7n-q551gr-miGvSAu6u8Kt9_kH93tmQ-bcEjx2jSiGPucMKuhf3JSxteSoniaZ1K_dz9jQwqmveZcBw"
root_dir = "D:/d55146e8-fc86-44dd-95db-7191fdca7f30"

client = BucketApiClient(token=token)

# access dataset bucket
bucket = client.buckets.get_dataset(dataset_id)

# for file in bucket.ls(prefix="sub-004"):
for file in bucket.ls():
    # 保存文件到本地
    path = os.path.join(root_dir, file.name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(file.name, "Exists")
        continue

    file_handle = bucket.get_file(file.name)
    file_content = file_handle.get_content(progress=True)
    with open(path, 'wb') as f:
        f.write(file_content)
    print(file.name, "Done")
