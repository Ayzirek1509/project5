import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_gaussian_noise_color(image, mean=0, sigma=20):
    """
    Rangli tasvirga Gauss shovqin qo'shish (test uchun).
    mean  – o'rtacha qiymat
    sigma – dispersiya (qanchalik katta bo'lsa, shovqin shunchalik kuchli)
    """
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def main():
    # 1) Rangli tasvirni o'qish (BGR format)
    img = cv2.imread('img_19.png')  # <-- bu yerga o'z rasm nomingni yoz
    if img is None:
        print("Rasm topilmadi! Fayl nomi yoki yo‘lini tekshir.")
        return

    # 2) Test uchun shovqin qo'shamiz.
    #    Agar sendagi rasm allaqachon shovqinli/xira bo'lsa:
    #    noisy = img  deb yozib qo'ysang bo'ladi.
    noisy = add_gaussian_noise_color(img, sigma=25)
    # noisy = img   # <-- agar tayyor shovqinli rasm bo'lsa, shu qatordan foydalan

    # 3) NON-LOCAL MEANS ALGORITHM (rangli tasvir uchun)
    # h va hColor – filtrlash kuchi, 5–15 oralig'ida o'ynab ko'r
    denoised = cv2.fastNlMeansDenoisingColored(
        noisy,                 # kirish rasm
        None,                  # chiqish (None bo'lsa, yangi rasm qaytaradi)
        h=10,                  # yorug'lik kanali uchun filtrlash kuchi
        hColor=10,             # rang kanali uchun filtrlash kuchi
        templateWindowSize=7,  # taqqoslash oynasi (odatda 7)
        searchWindowSize=21    # izlash oynasi (odatda 21)
    )

    # 4) Matplotlib RGB format bilan ishlaydi, BGR -> RGB konvert qilamiz
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    # 5) Natijalarni yonma-yon ko'rsatish
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_rgb)
    plt.title('Noisy (shovqinli)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_rgb)
    plt.title('Denoised (Non-Local Means)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 6) Natijani faylga saqlash
    cv2.imwrite('img_19.png', noisy)
    cv2.imwrite('color_denoised_nlm.png', denoised)
    print("color_noisy.png va color_denoised_nlm.png saqlandi.")


if __name__ == "__main__":
    main()
