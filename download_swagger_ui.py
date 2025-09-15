import requests
import os


def download_swagger_ui():
    # 创建 static 目录
    if not os.path.exists('static'):
        os.makedirs('static')

    files = [
        ('https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js', 'swagger-ui-bundle.js'),
        ('https://unpkg.com/swagger-ui-dist@5/swagger-ui.css', 'swagger-ui.css'),
        ('https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js', 'swagger-ui-standalone-preset.js')
    ]

    print("开始下载 Swagger UI 文件...")
    for url, filename in files:
        try:
            print(f"下载 {filename}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(f'static/{filename}', 'wb') as f:
                f.write(response.content)
            print(f"✓ {filename} 下载完成")
        except Exception as e:
            print(f"✗ 下载 {filename} 失败: {e}")
    print("文件下载完成！")


if __name__ == '__main__':
    download_swagger_ui()
