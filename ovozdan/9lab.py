import cv2
def apply_median_filter(image_path, kernel_size=5):
    # Rasmni o'qish
    img = cv2.imread(image_path)

    # Mediana filtrini qo'llash
    median_filtered = cv2.medianBlur(img, kernel_size)

    # Natijalarni ko'rsatish
    cv2.imshow('Asl rasm', img)
    cv2.imshow(f'Mediana filtri (kernel = {kernel_size})', median_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Natijani saqlash
    cv2.imwrite('img_3.png', median_filtered)

# Dastur ishga tushirish
apply_median_filter('img_3.png', kernel_size=7)
