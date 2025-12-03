import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_salt_pepper_noise(image, amount=0.2, s_vs_p=0.5):
    """
    Salt & pepper (oq va qora nuqtalar) shovqin qo'shish funksiyasi.
    amount  – shovqin miqdori (0..1 oralig'ida)
    s_vs_p – salt (oq) va pepper (qora) nisbatini boshqaradi
    """
    noisy = image.copy()

    # Umumiy pixellar soni
    num_pixels = image.size

    # Oq nuqtalar (salt) soni
    num_salt = int(amount * num_pixels * s_vs_p)
    # Qora nuqtalar (pepper) soni
    num_pepper = int(amount * num_pixels * (1 - s_vs_p))

    # Salt (oq nuqtalar)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper (qora nuqtalar)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy


def main():
    # 1) Rasmni o'qish (kulrang holatda)
    # Fayl nomini o'zingiznikiga moslab o'zgartiring, masalan: 'lena.png'
    img = cv2.imread('img_9.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Rasm topilmadi! Fayl nomi va yo'lini tekshiring.")
        return

    # 2) Salt & pepper shovqin qo'shamiz
    noisy_img = add_salt_pepper_noise(img, amount=0.2, s_vs_p=0.5)

    # 3) Median filtr bilan shovqinni tozalaymiz
    # 5x5 o'lchamli yadro (kernel)
    filtered_img = cv2.medianBlur(noisy_img, 5)

    # 4) Natijani ko'rsatish (Original – Filtered yonma-yon)
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_img, cmap='gray', vmin=0, vmax=255)
    plt.title('Filtered')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 5) Natijalarni faylga saqlash (xohlasangiz)
    cv2.imwrite('img_9.png', noisy_img)
    cv2.imwrite('img_9.png', filtered_img)
    print("img_3.png va img_9.png saqlandi.")


if __name__ == "__main__":
    main()
