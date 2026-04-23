import os
import platform
import zipfile
import urllib.request


def download_ffmpeg():
    system = platform.system().lower()
    if system == 'windows':
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        zip_path = "ffmpeg.zip"
        extract_dir = "ffmpeg"

        print("Downloading ffmpeg...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Move ffmpeg bin path to system path
        bin_path = None
        for root, dirs, files in os.walk(extract_dir):
            if 'ffmpeg.exe' in files:
                bin_path = root
                break

        if bin_path:
            os.environ["PATH"] += os.pathsep + bin_path
            print(f"ffmpeg installed and path added: {bin_path}")
        else:
            print("ffmpeg not found after extraction")

        os.remove(zip_path)
    else:
        print("Please install ffmpeg manually for non-Windows systems (Linux/macOS).")

if __name__ == "__main__":
    download_ffmpeg()
