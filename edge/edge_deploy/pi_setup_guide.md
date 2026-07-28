# 🍓 Guide de configuration pas à pas : Raspberry Pi (Premier démarrage)

Ce guide vous accompagne étape par étape pour préparer votre Raspberry Pi depuis votre PC Windows, activer le Wi-Fi et SSH, vous connecter à distance et installer l'environnement nécessaire pour faire tourner vos modèles ONNX ECG.

---

## 📦 Matériel nécessaire

1. **Raspberry Pi** (Pi 4 ou Pi 5)
2. **Carte MicroSD** (16 Go ou 32 Go+) + Lecteur de carte MicroSD pour votre PC
3. **Alimentation USB-C** (officielle recommandée)
4. **Réseau Wi-Fi** (le même réseau Wi-Fi que votre PC Windows)

---

## 🚀 ÉTAPE 1 : Télécharger Raspberry Pi Imager sur PC

1. Sur votre PC Windows, ouvrez votre navigateur et allez sur :  
   👉 **[https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)**
2. Téléchargez et installez **Raspberry Pi Imager for Windows**.
3. Insérez votre carte MicroSD dans le lecteur de carte de votre PC.

---

## 💾 ÉTAPE 2 : Flasher l'OS 64-bit avec pré-configuration Wi-Fi & SSH

> [!IMPORTANT]
> **Pourquoi le 64-bit ?**  
> L'OS **Raspberry Pi OS (64-bit)** est obligatoire pour exécuter ONNX Runtime et bénéficier des instructions vectorielles ARM NEON 64-bit.

1. Lancez **Raspberry Pi Imager** sur Windows.
2. **Choose Device (Périphérique) :** Sélectionnez `Raspberry Pi 4` ou `Raspberry Pi 5`.
3. **Choose OS (Système d'exploitation) :**  
   - Cliquez sur `Raspberry Pi OS (other)` → Choisissez **`Raspberry Pi OS (64-bit)`** (Raspberry Pi OS with desktop or Lite 64-bit).
4. **Choose Storage (Stockage) :** Sélectionnez votre carte MicroSD.
5. Cliquez sur **NEXT** (Suivant).
6. Une fenêtre apparaît : **"Would you like to apply OS customization settings?"**  
   👉 Cliquez sur **EDIT SETTINGS** (Modifier les paramètres).

### ⚙️ Paramètres de personnalisation (À configurer impérativement) :

- **General (Général) :**
  - [x] **Set hostname :** `raspberrypi` (ou `ecg-edge`)
  - [x] **Set username and password :**  
    - Username : `taher` (ou `pi`)  
    - Password : *(choisissez un mot de passe dont vous vous souviendrez)*
  - [x] **Configure wireless LAN :**  
    - SSID : *(nom de votre réseau Wi-Fi)*  
    - Password : *(clé Wi-Fi)*  
    - Wireless LAN country : `MA` (Maroc) ou `FR`
  - [x] **Set locale settings :** Time zone = `Africa/Casablanca`

- **Services :**
  - [x] **Enable SSH**  
  - Sélectionnez **`Use password authentication`**

7. Cliquez sur **SAVE** (Enregistrer) → Puis cliquez sur **YES** pour lancer le flashage.  
8. Attendez la fin de l'écriture et de la vérification (~3-5 minutes), puis retirez la carte MicroSD du PC.

---

## 🔌 ÉTAPE 3 : Premier démarrage du Raspberry Pi

1. Insérez la carte MicroSD dans le port MicroSD du Raspberry Pi.
2. Branchez le câble d'alimentation USB-C au Raspberry Pi.
3. Les LEDs sur la carte vont clignoter (rouge = alimentation, verte = activité MicroSD).
4. **Patientez 2 minutes** le temps que le Raspberry Pi démarre et se connecte automatiquement à votre Wi-Fi.

---

## 💻 ÉTAPE 4 : Se connecter au Raspberry Pi depuis PowerShell (Windows)

Pas besoin d'écran ni de clavier branché au Raspberry Pi ! Vous allez vous y connecter directement depuis votre terminal PowerShell Windows via **SSH**.

1. Sur votre PC Windows, ouvrez **PowerShell**.
2. Tapez la commande suivante (remplacez `taher` par le nom d'utilisateur choisi à l'étape 2) :

```powershell
ssh taher@raspberrypi.local
```

3. Lors de la première connexion, le système affiche un message de sécurité :  
   `Are you sure you want to continue connecting (yes/no/[fingerprint])?`  
   👉 Tapez **`yes`** puis faites `Entrée`.
4. Saisissez votre mot de passe et appuyez sur `Entrée`.  
   *(Remarque : rien ne s'affiche à l'écran pendant que vous tapez le mot de passe, c'est normal !)*

🎉 **Félicitations !** Vous êtes maintenant connecté au terminal de votre Raspberry Pi depuis votre PC Windows. Vous verrez une invite comme : `taher@raspberrypi:~ $`.

---

## 🛠️ ÉTAPE 5 : Mettre à jour le système et préparer Python

Une fois connecté en SSH sur le Raspberry Pi, exécutez ces commandes :

```bash
# 1. Mettre à jour les paquets système
sudo apt update && sudo apt upgrade -y

# 2. Installer Python 3, pip, venv et Git
sudo apt install -y python3-pip python3-venv git

# 3. Vérifier que vous êtes bien sur un OS 64-bit (aarch64)
uname -m
# Doit afficher : aarch64
```

---

## 📦 ÉTAPE 6 : Cloner votre projet et installer ONNX Runtime sur le Pi

Toujours dans le terminal SSH du Raspberry Pi :

```bash
# 1. Cloner votre dépôt Git
git clone https://github.com/votre-compte/Vers-une-chaine-ECG-multimodale-deployable-sur-edge-fondations-SMHFL-.git ecg-edge
cd ecg-edge

# 2. Créer l'environnement virtuel Python
python3 -m venv venv
source venv/bin/activate

# 3. Installer ONNX Runtime et les dépendances légères
pip install --upgrade pip
pip install onnxruntime numpy scipy pandas
```

---

## 🧪 ÉTAPE 7 : Tester l'inférence ONNX sur le Raspberry Pi

Vérifiez qu'ONNX Runtime fonctionne sur le Pi :

```bash
python3 -c "import onnxruntime as ort; import numpy as np; print('ONNX Runtime version:', ort.__version__); print('Providers:', ort.get_available_providers())"
```

Vous devez obtenir :
```text
ONNX Runtime version: 1.18.x (ou supérieure)
Providers: ['CPUExecutionProvider']
```

Ensuite, vous pourrez exécuter les benchmarks directement sur le Pi avec le modèle quantifié INT8 :

```bash
python3 -m edge.src.benchmark_onnx --model-path edge/models/ecg_model_int8_dynamic.onnx
```

---

## 💡 Astuces & Dépannage

- **Si `ssh taher@raspberrypi.local` ne trouve pas le Pi :**
  - Assurez-vous que votre PC et le Pi sont sur le même réseau Wi-Fi.
  - Allez sur l'interface de votre box Wi-Fi (ex: `192.168.1.1`) pour trouver l'adresse IP attribuée au Pi (ex: `192.168.1.45`), puis faites : `ssh taher@192.168.1.45`.
- **Pour éteindre proprement le Pi :**  
  `sudo shutdown -h now`
