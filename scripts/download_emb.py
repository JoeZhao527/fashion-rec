import requests
import os

def download_file_from_google_drive(file_id, destination, url):
    session = requests.Session()

    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    feat_dir = "./feature"
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    f_name = 'dino_image_emb.npy'
    print(f"Downlaoding {f_name}...")
    download_file_from_google_drive(
        'dino_image_emb.npy', os.path.join(feat_dir, f_name),
        url="https://drive.usercontent.google.com/download?confirm=xxx&id=1BACmF4PBhy-RmgSsj0GgZjJgcPZUXfND"
    )

    f_name = 'glove_text_emb.npy'
    print(f"Downlaoding {f_name}...")
    download_file_from_google_drive(
        'glove_text_emb.npy', os.path.join(feat_dir, f_name),
        url="https://drive.usercontent.google.com/download?confirm=xxx&id=1-w1A5_wPs1ROu5gRXtqylPtmFG2eXDpX"
    )

    print(f"File downloaded to {feat_dir}")
