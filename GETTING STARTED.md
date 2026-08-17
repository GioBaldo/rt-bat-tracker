# Guida all'installazione di Conda su Debian 13.5 e setup di rt-bat-tracker

Questa guida fornisce le istruzioni passo-passo per installare **Miniconda** (o Anaconda) su una distribuzione **Linux Debian 13.5** e configurare il pacchetto/repository **rt-bat-tracker**.

> **Nota sul percorso:** Se il repository si trova su un sistema Windows locale (`C:\Users\gioba\OneDrive\Documenti\GitHub\rt-bat-tracker`), sono descritte due opzioni principali:
> 1. Installazione via **WSL2** (Windows Subsystem for Linux) che accede direttamente al file system di Windows.
> 2. Installazione su una macchina **Debian 13.5 nativa/remota**, clonando il repository da GitHub o trasferendo i file.

---

## Parte 1: Installazione di Miniconda su Debian 13.5

Miniconda è una versione leggera di Anaconda che include solo Conda, Python e un piccolo insieme di pacchetti utili.

### 1. Aggiornare i pacchetti di sistema
Apri il terminale Debian ed esegui:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential
```

### 2. Scaricare lo script di installazione di Miniconda
Scarica l'ultimo installer per Linux a 64-bit:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### 3. Eseguire l'installer
Avvia l'installazione guidata:
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
* Premi `Invio` per scorrere i termini di licenza e digita `yes` per accettare.
* Conferma il percorso di installazione predefinito (di solito `~/miniconda3`).
* Quando viene chiesto se inizializzare Conda (`Do you wish to update your shell profile to automatically initialize conda?`), digita **`yes`**.

### 4. Attivare l'ambiente Conda
Per rendere effettive le modifiche nella sessione corrente:
```bash
source ~/.bashrc
```

Verifica la corretta installazione con:
```bash
conda --version
```

---

## Parte 2: Preparazione dell'Ambiente Virtuale

È raccomandabile creare un ambiente Conda dedicato per il progetto `rt-bat-tracker`.

```bash
# Creazione dell'ambiente (puoi specificare la versione di Python richiesta dal progetto)
conda create -n bat-tracker python=3.10 -y

# Attivazione dell'ambiente
conda activate bat-tracker
```

---

## Parte 3: Installazione di `rt-bat-tracker`



### Opzione B: Uso di Debian Nativo o Server Remoto (Clonazione via Git)

Se Debian è una macchina distinta e il repository è pubblicato su GitHub:

1. **Clona il repository GitHub:**
   ```bash
   cd ~
   git clone https://github.com/tuo-utente/rt-bat-tracker.git
   cd rt-bat-tracker
   ```
   *(Sostituisci l'URL con l'indirizzo reale del repository GitHub).*

2. **Installa il pacchetto:**
   ```bash
   pip install -e .
   ```

---


## Parte 4: Gestione delle Dipendenze

A seconda della struttura del repository `rt-bat-tracker`:

* **Se è presente un file `environment.yml`:**
  ```bash
  conda env create -f environment.yml
  conda activate rt-bat-tracker
  ```

* **Se è presente un file `requirements.txt`:**
  ```bash
  pip install -r requirements.txt
  ```

* **Se è presente `setup.py` o `pyproject.toml`:**
  ```bash
  pip install -e .
  ```

---

## Verifica dell'Installazione

Verifica che il pacchetto sia installato correttamente ed elencato tra i pacchetti Python dell'ambiente attivo:

```bash
pip list | grep rt-bat-tracker
```

In alternativa, apri l'interprete Python per verificare l'importazione:
```python
python -c "import rt_bat_tracker; print('Installazione completata con successo!')"
```
