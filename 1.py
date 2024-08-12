import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk
import tkinter as tk
from tkinter import messagebox
from tkinter import Button
from PIL import Image, ImageTk

cascara_ojo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def redimensionar_ojos(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ojos = cascara_ojo.detectMultiScale(gris, scaleFactor=1.4, minNeighbors=5, minSize=(45, 45))

    avg_ew = int(np.mean([ew for (ex, ey, ew, eh) in ojos]) * 1.35)
    avg_eh = int(np.mean([eh for (ex, ey, ew, eh) in ojos]) * 1.35)

    for (ex, ey, ew, eh) in ojos:
        centro_x, centro_y = ex + ew // 2, ey + eh // 2

        nuevo_ew, nuevo_eh = avg_ew, avg_eh
        nuevo_ex, nuevo_ey = centro_x - nuevo_ew // 2, centro_y - nuevo_eh // 2

        nuevo_ex = max(0, nuevo_ex)
        nuevo_ey = max(0, nuevo_ey)
        nuevo_ew = min(nuevo_ew, img.shape[1] - nuevo_ex)
        nuevo_eh = min(nuevo_eh, img.shape[0] - nuevo_ey)

        ojo_original = img[ey:ey + eh, ex:ex + ew]

        ojo_ampliado = cv2.resize(ojo_original, (nuevo_ew, nuevo_eh))

        mascara = np.zeros_like(ojo_ampliado)
        h, w, _ = mascara.shape
        cv2.ellipse(mascara, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, (255, 255, 255), -1)

        ojo_ampliado = cv2.bitwise_and(ojo_ampliado, mascara)

        roi = img[nuevo_ey:nuevo_ey + nuevo_eh, nuevo_ex:nuevo_ex + nuevo_ew]
        ojo_ampliado_gris = cv2.cvtColor(ojo_ampliado, cv2.COLOR_BGR2GRAY)
        _, mascara_inv = cv2.threshold(ojo_ampliado_gris, 1, 255, cv2.THRESH_BINARY_INV)
        roi_bg = cv2.bitwise_and(roi, roi, mask=mascara_inv)
        ojo_final = cv2.add(roi_bg, ojo_ampliado)

        img[nuevo_ey:nuevo_ey + nuevo_eh, nuevo_ex:nuevo_ex + nuevo_ew] = ojo_final

    return img

def ojos_caricatura(img, aumentado):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascara_ojo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    ojos = cascara_ojo.detectMultiScale(gris, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    avg_ew = int(np.mean([ew for (ex, ey, ew, eh) in ojos]) * aumentado)
    avg_eh = int(np.mean([eh for (ex, ey, ew, eh) in ojos]) * aumentado)


    for (ex, ey, ew, eh) in ojos:
        centro_x, centro_y = ex + ew // 2, ey + eh // 2

        nuevo_ew, nuevo_eh = avg_ew, avg_eh
        nuevo_ex, nuevo_ey = centro_x - nuevo_ew // 2, centro_y - nuevo_eh // 2

        nuevo_ex = max(0, nuevo_ex)
        nuevo_ey = max(0, nuevo_ey)
        nuevo_ew = min(nuevo_ew, img.shape[1] - nuevo_ex)
        nuevo_eh = min(nuevo_eh, img.shape[0] - nuevo_ey)

        ojo_original = img[ey:ey + eh, ex:ex + ew]

        ojo_ampliado = cv2.resize(ojo_original, (nuevo_ew, nuevo_eh))

        mascara = np.zeros_like(ojo_ampliado)
        h, w, _ = mascara.shape
        cv2.ellipse(mascara, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, (255, 255, 255), -1)

        ojo_ampliado = cv2.bitwise_and(ojo_ampliado, mascara)

        roi = img[nuevo_ey:nuevo_ey + nuevo_eh, nuevo_ex:nuevo_ex + nuevo_ew]
        ojo_ampliado_gris = cv2.cvtColor(ojo_ampliado, cv2.COLOR_BGR2GRAY)
        _, mascara_inv = cv2.threshold(ojo_ampliado_gris, 1, 255, cv2.THRESH_BINARY_INV)
        roi_bg = cv2.bitwise_and(roi, roi, mask=mascara_inv)
        ojo_final = cv2.add(roi_bg, ojo_ampliado)

        datos = np.float32(ojo_final).reshape((-1, 3))
        criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, etiqueta, centro = cv2.kmeans(datos, 3, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
        centro = np.uint8(centro)
        resultado = centro[etiqueta.flatten()]
        resultado = resultado.reshape(ojo_final.shape)

        img[nuevo_ey:nuevo_ey + nuevo_eh, nuevo_ex:nuevo_ex + nuevo_ew] = resultado

    return img

def convertir_a_caricatura(img):
    valor_difuminado = 7
    linea = 23
    k = 8
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gris_difuminado = cv2.medianBlur(gris, valor_difuminado)

    bordes = cv2.adaptiveThreshold(gris_difuminado, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, linea, valor_difuminado)

    piel_bajo = np.array([0, 48, 80])
    piel_alto = np.array([20, 255, 255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mascara = cv2.inRange(hsv, piel_bajo, piel_alto)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contorno = max(contornos, key=cv2.contourArea)

    datos = np.float32(img).reshape((-1, 3))
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, etiqueta, centro = cv2.kmeans(datos, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    centro = np.uint8(centro)
    resultado = centro[etiqueta.flatten()]
    resultado = resultado.reshape(img.shape)

    mascara_figura = np.zeros(resultado.shape[:2], dtype=np.uint8)
    cv2.drawContours(mascara_figura, [contorno], -1, 255, -1)

    color_figura = np.mean(resultado[mascara_figura == 255], axis=0)
    resultado[mascara_figura == 255] = color_figura

    resultado_final = cv2.bitwise_and(resultado, resultado, mask=bordes)
    return resultado_final


def detectar_y_dibujar_contornos(img, valor_difuminado, linea):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gris_difuminado = cv2.GaussianBlur(gris, (valor_difuminado, valor_difuminado), 0)


    bordes = cv2.adaptiveThreshold(gris_difuminado, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, linea, valor_difuminado)

    return bordes


def proceso_final(img):

    bordes = detectar_y_dibujar_contornos(img, 7, 15)


    imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, imagen_umbralizada = cv2.threshold(imagen_gris, 100, 255, cv2.THRESH_BINARY)


    contornos, _ = cv2.findContours(imagen_umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contornos_ojos = []
    contornos_boca = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if 100 < area < 2000:
            contornos_ojos.append(contorno)
        elif 2000 < area < 5000:
            contornos_boca.append(contorno)


    for contorno in contornos_ojos:
        cv2.drawContours(img, [contorno], -1, (0, 0, 0), 2)
    for contorno in contornos_boca:
        cv2.drawContours(img, [contorno], -1, (0, 0, 0), 2)

    resultado_final = cv2.bitwise_and(img, img, mask=bordes)
    return resultado_final


def procesar_imagen():
    ruta_imagen = filedialog.askopenfilename()

    if ruta_imagen:
        img = cv2.imread(ruta_imagen)
        cv2.imshow('Original', img)

        for i in range(3):
            imagen = convertir_a_caricatura(img)
            imagen = proceso_final(imagen)
            imagen = proceso_final(imagen)
        cv2.imshow('Caricaturización', imagen)
        cv2.waitKey(0)

        messagebox.showinfo("Información", "Ahora, ¿dónde quieres guardar tu caricatura?")
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".jpg")
        if ruta_guardado:
            cv2.imwrite(ruta_guardado, imagen)
            print(f"La caricatura se ha guardado como '{ruta_guardado}'")
        else:
            print("No se seleccionó ninguna ruta para guardar la caricatura.")

    else:
        print("No se seleccionó ninguna imagen.")


def procesar_imagen_experimental():
    ruta_imagen = filedialog.askopenfilename()

    if ruta_imagen:
        img = cv2.imread(ruta_imagen)
        cv2.imshow('Original', img)

        imagen = redimensionar_ojos(img)
        imagen = convertir_a_caricatura(imagen)
        imagen = proceso_final(imagen)
        imagen = convertir_a_caricatura(imagen)
        imagen = proceso_final(imagen)
        
        cv2.imshow('Caricaturización', imagen)
        cv2.waitKey(0)

        messagebox.showinfo("Información", "Ahora, ¿dónde quieres guardar tu caricatura?")
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".jpg")
        if ruta_guardado:
            cv2.imwrite(ruta_guardado, imagen)
            print(f"La caricatura se ha guardado como '{ruta_guardado}'")
        else:
            print("No se seleccionó ninguna ruta para guardar la caricatura.")
    else:
        print("No se seleccionó ninguna imagen.")


def mostrar_mensaje_experimental():
    mensaje = ("Es un modo de prueba donde a la caricatura le resaltan los ojos, creando un efecto más caricaturizado\n"
               "Al ser modo experimental debe tener en cuenta lo siguiente:\n" \
               ".- Algunas sombras, posiciones de rostro e incluso píxeles de la imagen pueden generar falsos positivos\n" \
               ".- En ciertas ocasiones algunas formas en las zonas como Boca, Orejas o Nariz el método los toma como si fueran parámetros de los ojos\n"
               "aumentando su tamaño deformando la imagen\n"
               ".- Personas con lentes no son leídas\n" \
               ".- Este modo sigue las instrucciones del caricaturizado estándar (si no conoce favor de leerlas dando clic en el botón caricatura)\n"
               ".- En la carpeta del programa están imágenes de prueba\n" \
               "¿Estás seguro de que deseas continuar?")
    respuesta = messagebox.askyesno("Experimental", mensaje)
    if respuesta:
        procesar_imagen_experimental()


def mostrar_mensaje_caricatura():
    mensaje = ("Proceso que caricaturiza la imagen de una persona\n"
               "Para un óptimo resultado debe tener en cuenta lo siguiente:\n" \
               ".- La persona debe salir sola en la imagen\n" \
               ".- El fondo debe ser unicolor (una pared blanca por ejemplo)\n"
               ".- Evitar imágenes de baja resolución\n" \
               ".- La facción de la persona debe ser clara (evite fotos opacas y de cuerpo completo)\n"
               ".- Las medidas recomendadas son 500x500 (alto x ancho) aproximadamente (no menores de 400x400)\n" \
               ".- En la carpeta del programa están imágenes de prueba")
    messagebox.showinfo("Caricatura", mensaje)
    procesar_imagen()


root = Tk()
root.geometry("300x500")
root.title("Crear Caricatura")

imagen_fondo = Image.open("Proyecto.png")
imagen_fondo_tk = ImageTk.PhotoImage(imagen_fondo)

etiqueta_fondo = tk.Label(root, image=imagen_fondo_tk)
etiqueta_fondo.place(relwidth=1, relheight=1)

etiqueta_fondo.image = imagen_fondo_tk

boton_experimental = Button(root, text="Experimental", command=mostrar_mensaje_experimental, font=("Consolas", 10, "bold"))
boton_experimental.place(x=100, y=430, width=90, height=30)

boton_caricatura = Button(root, text="Crear Caricatura", command=mostrar_mensaje_caricatura, font=("Consolas", 10, "bold"))
boton_caricatura.place(x=74, y=397, width=151, height=30)

root.mainloop()

