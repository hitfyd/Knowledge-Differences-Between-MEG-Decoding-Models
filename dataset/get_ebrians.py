import os

from ebrains_drive import BucketApiClient

dataset_id = "d55146e8-fc86-44dd-95db-7191fdca7f30"
token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJLYU01NTRCM2RmMHBIamZYWi1aRl94bUUwMThPS1R0RkNjMjR3aVVqQmFvIn0.eyJleHAiOjE3MzI4MDYzMzUsImlhdCI6MTczMjc2MzEzNCwiYXV0aF90aW1lIjoxNzMyNzYyMDYyLCJqdGkiOiI0NjBlZDlhZC02ZDBhLTQ5ZDgtYjkyMi1mNDczYmM4OTMyMDkiLCJpc3MiOiJodHRwczovL2lhbS5lYnJhaW5zLmV1L2F1dGgvcmVhbG1zL2hicCIsImF1ZCI6WyJqdXB5dGVyaHViLWpzYyIsInh3aWtpIiwidGVhbSIsImdyb3VwIl0sInN1YiI6IjJiYzcxNjIwLWI3NzktNGM3NS1hNGE4LWFiZTkxMTIxODBkNSIsInR5cCI6IkJlYXJlciIsImF6cCI6Imp1cHl0ZXJodWIiLCJzZXNzaW9uX3N0YXRlIjoiYmI2OTNhYjYtNjc2NS00OGJjLTg4ZjItNzk5MGNkMjJmMzM2IiwiYWxsb3dlZC1vcmlnaW5zIjpbImh0dHBzOi8vanVweXRlcmh1Yi5hcHBzLmpzYy5oYnAuZXUvIiwiaHR0cHM6Ly9sYWIuZWJyYWlucy5ldS8iLCJodHRwczovL2xhYi5qc2MuZWJyYWlucy5ldS8iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIl19LCJzY29wZSI6ImNvbGxhYi5kcml2ZSBwcm9maWxlIG9mZmxpbmVfYWNjZXNzIGNsYi53aWtpLndyaXRlIGVtYWlsIHJvbGVzIG9wZW5pZCBncm91cCBjbGIud2lraS5yZWFkIHRlYW0iLCJzaWQiOiJiYjY5M2FiNi02NzY1LTQ4YmMtODhmMi03OTkwY2QyMmYzMzYiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmFtZSI6Illvbmdkb25nIEZhbiIsIm1pdHJlaWQtc3ViIjoiMzIyNjAzMjgyODk3NjI3NCIsInByZWZlcnJlZF91c2VybmFtZSI6ImhpdGZ5ZCIsImdpdmVuX25hbWUiOiJZb25nZG9uZyIsImZhbWlseV9uYW1lIjoiRmFuIiwiZW1haWwiOiJoaXRmeWRAaGl0LmVkdS5jbiJ9.S18hP2Jj6COUuV_EYaZOY_gJXQyeVu2IxwE4uIvttIbD0Kxb0D2ad6L1iw1-YpEh1kmnDkDushavkS52S9qC49R0rCaaLU34sSokgZswXpM2OsAjR-eGLMMNNLc4BT3GL9eYdoOb7eW6qf1aA5-qlA7AtDlyxdZjdLWqJ7LFSLsyGy5gavgkE03RRDWNoAcYU6VTlvmLXfXgqhIVX_-mFpwCnDKWVZ6JsxODRkeLUXuEfoEoSMGV4TwHVRPVKmUGXxkei2UEE9u91i3v7DBgYGE29LEUHjc7eF25z59xwpsy6ZcVhMSFXqomjdwiqQvKGY-Io7xBszY56PK03-3wtA"
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

    # if "rest" not in file.name:
    #     print(file.name, "Skips")
    #     continue

    file_handle = bucket.get_file(file.name)
    file_content = file_handle.get_content(progress=True)
    with open(path, 'wb') as f:
        f.write(file_content)
    print(file.name, "Done")
