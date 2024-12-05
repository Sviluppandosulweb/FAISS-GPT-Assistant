import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import mimetypes
import os
import json
import time
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import logging
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Importazioni per il watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configurazione del logger per stampare i messaggi sulla console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

lock = threading.Lock()

# Configurazione del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_handler = None  # Verrà assegnato quando la finestra dei log verrà aperta

# Variabili globali per gli archivi e l'archivio corrente
archives = {}  # Dizionario per contenere gli archivi
current_archive_name = None  # Nome dell'archivio attualmente selezionato
watchdog_observer = None  # Osservatore per il monitoraggio delle modifiche ai file

# Variabili per i modelli
gpt_model_var = None
embedding_model_var = None
embeddings = None
llm = None
max_tokens_var = None  # Variabile per il massimo numero di token

# Percorso del file per memorizzare l'elenco degli archivi
archives_list_file = "archives_list.json"

# Creazione dell'interfaccia grafica
root = tk.Tk()
root.title("FAISS Vector Store & ChatGPT")

# Variabile per il menu a tendina degli archivi
archive_var = tk.StringVar()
archive_var.set('Nessun Archivio')

# Layout principale
main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10)

# Sezione per la selezione dei modelli
models_frame = tk.LabelFrame(main_frame, text="Selezione Modelli")
models_frame.pack(fill="x", pady=5)

# Opzioni per i modelli GPT
gpt_models = ['gpt-3.5-turbo', 'gpt-4']
gpt_model_var = tk.StringVar()
gpt_model_var.set(gpt_models[0])  # Imposta il modello GPT predefinito

# Menu a tendina per la selezione del modello GPT
gpt_model_label = tk.Label(models_frame, text="Seleziona Modello GPT:")
gpt_model_label.pack(side="left", padx=5, pady=5)

gpt_model_menu = tk.OptionMenu(models_frame, gpt_model_var, *gpt_models)
gpt_model_menu.pack(side="left", padx=5, pady=5)

# Opzioni per i modelli di embedding
embedding_models = ['text-embedding-ada-002']
embedding_model_var = tk.StringVar()
embedding_model_var.set(embedding_models[0])  # Imposta il modello di embedding predefinito

# Menu a tendina per la selezione del modello di embedding
embedding_model_label = tk.Label(models_frame, text="Seleziona Modello di Embedding:")
embedding_model_label.pack(side="left", padx=5, pady=5)

embedding_model_menu = tk.OptionMenu(models_frame, embedding_model_var, *embedding_models)
embedding_model_menu.pack(side="left", padx=5, pady=5)

# Variabile per il massimo numero di token
max_tokens_var = tk.IntVar()
max_tokens_var.set(8192)  # Imposta il limite predefinito (per gpt-3.5-turbo)

# Slider per impostare il massimo numero di token
max_tokens_label = tk.Label(models_frame, text="Massimo numero di token:")
max_tokens_label.pack(side="left", padx=5, pady=5)

max_tokens_slider = tk.Scale(models_frame, from_=1000, to=32000, orient='horizontal', variable=max_tokens_var)
max_tokens_slider.pack(side="left", padx=5, pady=5)

# Funzione per determinare se un file è binario
def is_binary_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            CHUNK_SIZE = 1024
            while True:
                chunk = file.read(CHUNK_SIZE)
                if not chunk:
                    break
                if b'\0' in chunk:
                    return True
        return False
    except Exception as e:
        logger.error(f"Errore durante la verifica se il file è binario: {str(e)}")
        return True

# Funzione per aggiornare i modelli quando la selezione cambia
def update_models(*args):
    global embeddings, llm
    # Aggiorna il modello di embedding
    embeddings = OpenAIEmbeddings(model=embedding_model_var.get())
    logger.info(f"Modello di embedding aggiornato a: {embedding_model_var.get()}")
    # Aggiorna il modello GPT
    llm = ChatOpenAI(model=gpt_model_var.get(), temperature=0)
    logger.info(f"Modello GPT aggiornato a: {gpt_model_var.get()}")
    # Salva le impostazioni del modello
    save_model_settings()

# Collega le variabili alla funzione di aggiornamento
gpt_model_var.trace('w', update_models)
embedding_model_var.trace('w', update_models)

# Funzione per salvare le impostazioni del modello GPT ed embedding
def save_model_settings():
    """
    Salva le impostazioni dei modelli GPT e embedding in un file 'config.json'.
    """
    config = {
        "gpt_model": gpt_model_var.get(),
        "embedding_model": embedding_model_var.get(),
        "max_tokens": max_tokens_var.get()
    }
    try:
        with open("config.json", "w") as config_file:
            json.dump(config, config_file)
        logger.info("Impostazioni dei modelli salvate con successo.")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio delle impostazioni dei modelli: {str(e)}")

# Funzione per caricare le impostazioni del modello GPT ed embedding
def load_model_settings():
    """
    Carica le impostazioni dei modelli GPT e embedding da un file 'config.json', se esiste.
    """
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
                gpt_model_var.set(config.get("gpt_model", gpt_models[0]))
                embedding_model_var.set(config.get("embedding_model", embedding_models[0]))
                max_tokens_var.set(config.get("max_tokens", 8192))
            logger.info("Impostazioni dei modelli caricate correttamente.")
        except Exception as e:
            logger.error(f"Errore durante il caricamento delle impostazioni dei modelli: {str(e)}")
    else:
        logger.info("Nessun file di configurazione trovato, impostazioni predefinite caricate.")

# Caricamento delle impostazioni del modello GPT ed embedding all'avvio
load_model_settings()
update_models()

# Sezione per la selezione dell'archivio
archive_frame = tk.LabelFrame(main_frame, text="Archivio")
archive_frame.pack(fill="x", pady=5)

# Menu a tendina per la selezione dell'archivio
archive_label = tk.Label(archive_frame, text="Seleziona Archivio:")
archive_label.pack(side="left", padx=5, pady=5)

archive_menu = tk.OptionMenu(archive_frame, archive_var, ())
archive_menu.pack(side="left", padx=5, pady=5)

# Pulsante per creare un nuovo archivio
button_new_archive = tk.Button(archive_frame, text="Crea Nuovo Archivio", command=lambda: create_new_archive())
button_new_archive.pack(side="left", padx=5, pady=5)

# Pulsante per caricare un archivio esistente
button_load_archive = tk.Button(archive_frame, text="Carica Archivio Esistente", command=lambda: load_existing_archive())
button_load_archive.pack(side="left", padx=5, pady=5)

# Pulsante per rinominare un archivio
button_rename_archive = tk.Button(archive_frame, text="Rinomina Archivio", command=lambda: rename_archive())
button_rename_archive.pack(side="left", padx=5, pady=5)

# Pulsante per eliminare un archivio
button_delete_archive = tk.Button(archive_frame, text="Elimina Archivio", command=lambda: delete_archive())
button_delete_archive.pack(side="left", padx=5, pady=5)

# Barra di progresso e label per tempo stimato (inizialmente nascosti)
progress_bar = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate", length=300)
progress_label = tk.Label(main_frame, text="Tempo stimato: --:--")

# Funzione per rinominare un archivio
def rename_archive():
    global current_archive_name
    if current_archive_name is None:
        messagebox.showerror("Errore", "Per favore, seleziona un archivio prima!")
        return

    new_name = simpledialog.askstring("Rinomina Archivio", "Inserisci il nuovo nome per l'archivio:")
    if new_name and new_name not in archives:
        archives[new_name] = archives.pop(current_archive_name)
        current_archive_name = new_name
        update_archive_selection()
        save_archives_list()
        messagebox.showinfo("Successo", f"Archivio rinominato con successo a '{new_name}'!")
    else:
        messagebox.showerror("Errore", "Nome non valido o già esistente!")

# Funzione per eliminare un archivio
def delete_archive():
    global current_archive_name
    if current_archive_name is None:
        messagebox.showerror("Errore", "Per favore, seleziona un archivio prima!")
        return

    delete_files = messagebox.askyesno("Elimina Archivio", "Vuoi anche eliminare i file FAISS associati?")
    if delete_files:
        archive = archives[current_archive_name]
        try:
            if os.path.exists(archive['faiss_index_path']):
                os.remove(archive['faiss_index_path'])
            if os.path.exists(archive['file_paths_path']):
                os.remove(archive['file_paths_path'])
            if os.path.exists(archive['timestamps_path']):
                os.remove(archive['timestamps_path'])
            if os.path.exists(archive['conversation_history_path']):
                os.remove(archive['conversation_history_path'])
            messagebox.showinfo("Successo", "File FAISS eliminati con successo!")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante l'eliminazione dei file: {str(e)}")

    # Elimina l'archivio dalla lista
    archives.pop(current_archive_name)
    current_archive_name = None
    update_archive_selection()
    save_archives_list()
    messagebox.showinfo("Successo", "Archivio eliminato con successo!")

# Funzione per caricare l'elenco degli archivi dall'archivio_list_file
def load_archives_list():
    """
    Carica l'elenco degli archivi dal file 'archives_list.json' se esiste.
    Se non esiste, chiede la cartella sorgente all'utente.
    """
    if os.path.exists(archives_list_file):
        try:
            with open(archives_list_file, 'r') as f:
                archives_info = json.load(f)
                for archive_name, paths in archives_info.items():
                    logger.info(f"Caricamento dell'archivio '{archive_name}'")
                    conversation_history_path = paths.get('conversation_history_path', os.path.join(os.path.dirname(paths['faiss_index_path']), "conversation_history.json"))
                    
                    # Verifica se la directory dei file sorgente esiste
                    source_directory = paths.get('source_directory')
                    if not source_directory or not os.path.exists(source_directory):
                        # Se la directory non esiste o non è definita, chiedi all'utente di selezionarla
                        source_directory = filedialog.askdirectory(title=f"Seleziona la cartella sorgente per '{archive_name}'")
                        if not source_directory:
                            messagebox.showerror("Errore", "Non è stata selezionata nessuna cartella sorgente. L'archivio non può essere caricato.")
                            continue
                    
                    # Aggiorna i dati dell'archivio con la nuova directory sorgente
                    archives[archive_name] = {
                        'faiss_index': None,
                        'file_paths': [],
                        'index_to_docstore_id': {},
                        'docstore': None,
                        'faiss_index_path': paths['faiss_index_path'],
                        'file_paths_path': paths['file_paths_path'],
                        'timestamps_path': paths.get('timestamps_path', ''),
                        'conversation_history': [],
                        'conversation_history_path': conversation_history_path,
                        'source_directory': source_directory,
                        'monitored_paths': paths.get('monitored_paths', [])
                    }
                    logger.info(f"Archivio '{archive_name}' caricato correttamente con directory sorgente: {source_directory}")
            logger.info("Archivi caricati correttamente.")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante il caricamento degli archivi: {str(e)}")
    else:
        logger.warning(f"Il file '{archives_list_file}' non esiste.")



# Funzione per salvare l'elenco degli archivi su 'archives_list.json'
def save_archives_list():
    """
    Salva l'elenco degli archivi nel file 'archives_list.json'.
    """
    try:
        archives_info = {}
        for archive_name, data in archives.items():
            archives_info[archive_name] = {
                'faiss_index_path': data['faiss_index_path'],
                'file_paths_path': data['file_paths_path'],
                'timestamps_path': data['timestamps_path'],
                'conversation_history_path': data['conversation_history_path'],
                'source_directory': data.get('source_directory', ''),
                'monitored_paths': data['monitored_paths']
            }
        with open(archives_list_file, 'w') as f:
            json.dump(archives_info, f)
        logger.info("Elenco degli archivi salvato con successo.")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio dell'elenco degli archivi: {str(e)}")

# Funzione per creare un nuovo archivio
def create_new_archive():
    """
    Crea un nuovo archivio chiedendo all'utente un nome e inizializza le sue strutture dati.
    """
    global current_archive_name
    archive_name = simpledialog.askstring("Crea Nuovo Archivio", "Inserisci un nome per il nuovo archivio:")
    if archive_name:
        if archive_name in archives:
            messagebox.showerror("Errore", f"Un archivio chiamato '{archive_name}' esiste già!")
            return
        # Chiedi dove salvare i file dell'archivio
        archive_dir = filedialog.askdirectory(title="Seleziona la cartella dove salvare l'archivio")
        if not archive_dir:
            messagebox.showwarning("Attenzione", "Creazione archivio annullata.")
            return

        # Chiedi dove si trovano i file da monitorare
        source_directory = filedialog.askdirectory(title="Seleziona la cartella con i file da monitorare")
        if not source_directory:
            messagebox.showwarning("Attenzione", "Creazione archivio annullata: nessuna cartella selezionata per i file.")
            return

        faiss_index_path = os.path.join(archive_dir, "faiss_index")
        file_paths_path = os.path.join(archive_dir, "file_paths.json")
        timestamps_path = os.path.join(archive_dir, "timestamps.json")
        conversation_history_path = os.path.join(archive_dir, "conversation_history.json")

        logger.info(f"Nuovo archivio creato con directory sorgente: {source_directory}")

        # Inizializza i dati dell'archivio
        archives[archive_name] = {
            'faiss_index': None,
            'file_paths': [],
            'index_to_docstore_id': {},
            'docstore': None,
            'faiss_index_path': faiss_index_path,
            'file_paths_path': file_paths_path,
            'timestamps_path': timestamps_path,
            'conversation_history': [],
            'conversation_history_path': conversation_history_path,
            'source_directory': source_directory,
            'monitored_paths': []
        }
        current_archive_name = archive_name
        update_archive_selection()
        save_archives_list()
        messagebox.showinfo("Successo", f"Archivio '{archive_name}' creato e selezionato con directory sorgente: '{source_directory}'!")
    else:
        messagebox.showerror("Errore", "Il nome dell'archivio non può essere vuoto!")

# Funzione per aggiornare il menu di selezione degli archivi
def update_archive_selection():
    """
    Aggiorna il menu a tendina degli archivi disponibili e seleziona l'archivio corrente.
    """
    archive_names = list(archives.keys())
    menu = archive_menu['menu']
    menu.delete(0, 'end')
    if archive_names:
        for name in archive_names:
            menu.add_command(label=name, command=lambda value=name: on_archive_select(value))
        archive_var.set(current_archive_name)
    else:
        archive_var.set('Nessun Archivio')

# Funzione chiamata quando un archivio viene selezionato dal menu a tendina
def on_archive_select(value):
    """
    Gestisce la selezione di un archivio dal menu a tendina.
    """
    global current_archive_name
    stop_watchdog()  # Ferma l'osservatore corrente prima di cambiare archivio
    current_archive_name = value
    archive_var.set(current_archive_name)
    # Carica l'indice FAISS e i percorsi dei file per l'archivio selezionato
    load_faiss_index(current_archive_name)
    # Carica la cronologia delle conversazioni
    load_conversation_history()
    # Avvia il monitoraggio dei file per l'archivio selezionato
    start_watchdog_for_current_archive()
    messagebox.showinfo("Archivio Selezionato", f"Archivio '{current_archive_name}' è ora attivo.")


# Funzione per indicizzare un singolo file nell'archivio corrente
def index_single_file(file_path):
    global current_archive_name
    if current_archive_name is None:
        messagebox.showerror("Errore", "Per favore, crea o seleziona un archivio prima!")
        return

    # Controlla se il file è all'interno di una directory .git
    if '.git' in file_path.split(os.sep):
        logger.info(f"File in directory .git '{file_path}' rilevato. Indicizzo solo nome e percorso.")
        content = f"File in directory .git: {os.path.basename(file_path)}"
    else:
        # Per i file non in .git
        if is_binary_file(file_path):
            logger.info(f"File binario '{file_path}' rilevato. Indicizzo solo nome e percorso.")
            content = f"File binario: {os.path.basename(file_path)}"
        else:
            # Leggi il contenuto del file di testo
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Errore durante l'indicizzazione del file '{file_path}': {str(e)}")
                    return
            except Exception as e:
                logger.error(f"Errore durante l'indicizzazione del file '{file_path}': {str(e)}")
                return

    try:
        # Embed del contenuto
        embedding = embeddings.embed_documents([content])[0]

        with lock:
            archive = archives[current_archive_name]
            if archive['faiss_index'] is None:
                d = len(embedding)
                archive['faiss_index'] = faiss.IndexFlatL2(d)

            archive['faiss_index'].add(np.array([embedding]))
            archive['file_paths'].append(file_path)

            if 'modification_timestamps' not in archive:
                archive['modification_timestamps'] = {}
            archive['modification_timestamps'][file_path] = os.path.getmtime(file_path)

            update_index_to_docstore(current_archive_name)
            logger.info(f"File '{file_path}' indicizzato con successo nell'archivio '{current_archive_name}'!")

            if file_path not in archive['monitored_paths']:
                archive['monitored_paths'].append(file_path)

    except Exception as e:
        if "rate limit" in str(e).lower():
            logger.warning("Rate limit raggiunto. Attesa di 60 secondi prima di riprovare.")
            time.sleep(60)
            index_single_file(file_path)
        else:
            logger.error(f"Errore durante l'indicizzazione del file '{file_path}': {str(e)}")

# Funzione per indicizzare un singolo file e aggiornare la barra di progresso
def index_single_file_with_progress(file_path, index, total_files, start_time):
    index_single_file(file_path)
    # Aggiorna la barra di progresso
    progress_bar['value'] += 1
    elapsed_time = time.time() - start_time
    files_processed = progress_bar['value']
    if files_processed > 0:
        estimated_total_time = (elapsed_time / files_processed) * total_files
        remaining_time = estimated_total_time - elapsed_time
        minutes, seconds = divmod(remaining_time, 60)
        progress_label.config(text=f"Tempo stimato: {int(minutes)} min {int(seconds)} sec")
    else:
        progress_label.config(text="Tempo stimato: calcolando...")

# Funzione per indicizzare più file con multithreading
def index_multiple_files(file_paths):
    """
    Indica una lista di file nell'archivio corrente usando il multithreading.
    """
    global current_archive_name
    if current_archive_name is None:
        messagebox.showerror("Errore", "Per favore, crea o seleziona un archivio prima!")
        return

    def index_files_thread():
        try:
            total_files = len(file_paths)
            if total_files == 0:
                messagebox.showwarning("Attenzione", "Nessun file selezionato per l'indicizzazione.")
                return

            # Mostra la barra di progresso e l'etichetta
            progress_bar['maximum'] = total_files
            progress_bar['value'] = 0
            progress_bar.pack(pady=5)
            progress_label.config(text="Tempo stimato: calcolando...")
            progress_label.pack(pady=5)

            start_time = time.time()

            # Usa ThreadPoolExecutor per indicizzare i file in parallelo
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i, file_path in enumerate(file_paths):
                    futures.append(executor.submit(index_single_file_with_progress, file_path, i, total_files, start_time))

                # Attendi il completamento di tutte le future
                for future in futures:
                    future.result()

            # Nascondi la barra di progresso e l'etichetta
            progress_bar.pack_forget()
            progress_label.pack_forget()

            messagebox.showinfo("Successo", f"File selezionati indicizzati con successo nell'archivio '{current_archive_name}'!")

            # Dopo l'indicizzazione, avvia o aggiorna il monitoraggio dei file
            start_watchdog_for_current_archive()

        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante l'indicizzazione dei file: {str(e)}")

    # Avvia il thread per l'indicizzazione
    threading.Thread(target=index_files_thread).start()

# Funzione per aggiornare il mapping index-to-docstore per un archivio
def update_index_to_docstore(archive_name):
    archive = archives[archive_name]
    archive['index_to_docstore_id'] = {i: str(i) for i in range(len(archive['file_paths']))}

    docstore_dict = {}
    for i, path in enumerate(archive['file_paths']):
        if '.git' in path.split(os.sep):
            content = f"File in directory .git: {os.path.basename(path)}"
        elif is_binary_file(path):
            content = f"File binario: {os.path.basename(path)}"
        else:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                content = f"Errore nella lettura del file: {str(e)}"
                logger.error(f"Errore nella lettura del file '{path}': {str(e)}")

        docstore_dict[str(i)] = Document(
            page_content=content,
            metadata={"source": path}
        )
    archive['docstore'] = InMemoryDocstore(docstore_dict)

# Funzione per salvare l'indice FAISS e i percorsi dei file per l'archivio corrente
def save_faiss_index():
    """
    Salva l'indice FAISS, i percorsi dei file e i timestamp per l'archivio corrente su disco.
    """
    global current_archive_name
    if current_archive_name is None:
        messagebox.showerror("Errore", "Per favore, crea o seleziona un archivio prima!")
        return
    try:
        archive = archives[current_archive_name]
        if archive['faiss_index'] is not None:
            # Salva l'indice FAISS
            faiss.write_index(archive['faiss_index'], archive['faiss_index_path'])
            # Salva i percorsi dei file
            with open(archive['file_paths_path'], 'w') as f:
                json.dump(archive['file_paths'], f)
            # Salva i timestamp dei file
            with open(archive['timestamps_path'], 'w') as f:
                json.dump(archive.get('modification_timestamps', {}), f)
            # Salva la cronologia delle conversazioni
            save_conversation_history()
            save_archives_list()
            messagebox.showinfo("Successo", f"Indice FAISS salvato con successo per l'archivio '{current_archive_name}'!")
        else:
            messagebox.showwarning("Attenzione", "Nessun indice da salvare!")

    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante il salvataggio dell'indice: {str(e)}")

# Funzione per caricare un indice FAISS e i percorsi dei file per un archivio
def load_faiss_index(archive_name):
    """
    Carica l'indice FAISS, i percorsi dei file e i timestamp per l'archivio specificato.
    """

    logger.info(f"Caricamento dell'indice FAISS per l'archivio '{archive_name}'")
    try:
        archive = archives[archive_name]
        # Carica l'indice FAISS
        if archive['faiss_index'] is None:
            if os.path.exists(archive['faiss_index_path']) and os.path.exists(archive['file_paths_path']):
                archive['faiss_index'] = faiss.read_index(archive['faiss_index_path'])
                with open(archive['file_paths_path'], 'r') as f:
                    archive['file_paths'] = json.load(f)
                # Carica i timestamp se il file esiste, altrimenti generali
                if os.path.exists(archive['timestamps_path']):
                    with open(archive['timestamps_path'], 'r') as f:
                        archive['modification_timestamps'] = json.load(f)
                else:
                    archive['modification_timestamps'] = {path: os.path.getmtime(path) for path in archive['file_paths']}

                # Aggiorna il mapping index-to-docstore
                update_index_to_docstore(archive_name)
                logger.info(f"Archivio '{archive_name}' caricato correttamente.")
            else:
                messagebox.showwarning("Attenzione", f"Indice FAISS o file_paths.json non trovati per l'archivio '{archive_name}'.")
    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante il caricamento dell'indice per l'archivio '{archive_name}': {str(e)}")

# Funzione per salvare la cronologia delle conversazioni
def save_conversation_history():
    global current_archive_name
    archive = archives[current_archive_name]
    conversation_path = archive['conversation_history_path']
    try:
        with open(conversation_path, 'w') as f:
            json.dump(archive['conversation_history'], f)
            logger.info(f"Cronologia delle conversazioni salvata per l'archivio '{current_archive_name}'.")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio della cronologia delle conversazioni: {str(e)}")

# Funzione per caricare la cronologia delle conversazioni
def load_conversation_history():
    """
    Carica la cronologia delle conversazioni per l'archivio corrente, verificando i file utilizzati.
    Se i file sono stati modificati, invalida la risposta e la rimuove dalla cronologia.
    """
    global current_archive_name
    archive = archives[current_archive_name]
    conversation_path = archive['conversation_history_path']

    if os.path.exists(conversation_path):
        try:
            with open(conversation_path, 'r') as f:
                conversation_history = json.load(f)
                # Verifica ogni elemento della cronologia
                valid_conversation_history = []
                for item in conversation_history:
                    # Controlla se i file utilizzati per generare la risposta sono stati modificati
                    file_modified = False
                    for file_path, timestamp in item['file_timestamps'].items():
                        # Se il file non esiste o il timestamp è cambiato, la risposta è obsoleta
                        if not os.path.exists(file_path) or os.path.getmtime(file_path) != timestamp:
                            file_modified = True
                            break
                    if not file_modified:
                        valid_conversation_history.append(item)
                    else:
                        logger.info(f"Risposta invalidata per la domanda '{item['question']}' a causa di modifiche nei file.")

                # Aggiorna la cronologia valida
                archive['conversation_history'] = valid_conversation_history
                logger.info(f"Cronologia delle conversazioni caricata per l'archivio '{current_archive_name}'.")
        except json.JSONDecodeError:
            logger.error(f"Errore nel decodificare il file di cronologia: '{conversation_path}'.")
            archive['conversation_history'] = []
        except Exception as e:
            logger.error(f"Errore durante il caricamento della cronologia delle conversazioni: {str(e)}")
            archive['conversation_history'] = []
    else:
        archive['conversation_history'] = []
        logger.info(f"Nessun file di cronologia trovato per l'archivio '{current_archive_name}', inizializzazione vuota.")

# Funzione per caricare tutti gli archivi esistenti all'avvio
def load_all_archives():
    """
    Carica tutti gli archivi esistenti dall'elenco degli archivi.
    """
    load_archives_list()
    update_archive_selection()

# Funzione per caricare un archivio esistente
def load_existing_archive():
    """
    Permette all'utente di caricare un archivio esistente selezionando i file 'faiss_index' e 'file_paths.json'.
    """
    try:
        # Seleziona il file 'faiss_index'
        faiss_index_path = filedialog.askopenfilename(title="Seleziona il file 'faiss_index'")
        if not faiss_index_path:
            messagebox.showwarning("Attenzione", "Caricamento annullato.")
            return

        # Seleziona il file 'file_paths.json'
        file_paths_path = filedialog.askopenfilename(title="Seleziona il file 'file_paths.json'")
        if not file_paths_path:
            messagebox.showwarning("Attenzione", "Caricamento annullato.")
            return

        # Chiedi il nome dell'archivio
        archive_name = simpledialog.askstring("Nome Archivio", "Inserisci un nome per questo archivio:")
        if not archive_name:
            messagebox.showerror("Errore", "Il nome dell'archivio non può essere vuoto!")
            return
        if archive_name in archives:
            messagebox.showerror("Errore", f"Un archivio chiamato '{archive_name}' esiste già!")
            return

        # Chiedi la directory dei file sorgente
        source_directory = filedialog.askdirectory(title="Seleziona la cartella con i file da monitorare")
        if not source_directory:
            messagebox.showwarning("Attenzione", "Caricamento annullato: nessuna cartella selezionata per i file.")
            return

        timestamps_path = os.path.join(os.path.dirname(faiss_index_path), "timestamps.json")
        conversation_history_path = os.path.join(os.path.dirname(faiss_index_path), "conversation_history.json")

        # Inizializza i dati dell'archivio
        archives[archive_name] = {
            'faiss_index': None,
            'file_paths': [],
            'index_to_docstore_id': {},
            'docstore': None,
            'faiss_index_path': faiss_index_path,
            'file_paths_path': file_paths_path,
            'timestamps_path': timestamps_path,
            'conversation_history': [],
            'conversation_history_path': conversation_history_path,
            'source_directory': source_directory,
            'monitored_paths': []
        }

        # Log per il debug
        logger.info(f"Archivio '{archive_name}' caricato con directory sorgente: {source_directory}")

        global current_archive_name
        current_archive_name = archive_name
        update_archive_selection()
        save_archives_list()

        # Carica l'indice e i percorsi dei file
        load_faiss_index(archive_name)

        # Carica la cronologia delle conversazioni
        load_conversation_history()

        # Avvia il monitoraggio dei file
        start_watchdog_for_current_archive()
        messagebox.showinfo("Successo", f"Archivio '{archive_name}' caricato e selezionato con directory sorgente: '{source_directory}'!")

    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante il caricamento dell'archivio: {str(e)}")
        
# Imposta un logger
logger = logging.getLogger(__name__)

# Funzione per calcolare il checksum del file (MD5)
def calculate_file_checksum(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except Exception as e:
        logger.error(f"Errore nel calcolo del checksum per il file {file_path}: {str(e)}")
        return None
    return hash_md5.hexdigest()

# Handler per gli eventi del file system
class FileChangeHandler(FileSystemEventHandler):
    """
    Gestisce gli eventi di modifica dei file monitorati.
    Utilizza un controllo basato sul checksum per rilevare variazioni di contenuto.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_checksums = {}  # Dizionario per tenere traccia dei checksum precedenti

    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path

            # Calcoliamo il checksum attuale del file
            current_checksum = calculate_file_checksum(file_path)
            if current_checksum is None:
                logger.error(f"Impossibile calcolare il checksum per {file_path}. Modifica ignorata.")
                return

            last_checksum = self.last_checksums.get(file_path, "")

            logger.info(f"Checksum attuale: {current_checksum}, Ultimo checksum registrato: {last_checksum}")

            # Controlliamo se il checksum è cambiato
            if current_checksum != last_checksum:
                logger.info(f"Modifica rilevata nel file: {file_path}, checksum diverso rilevato.")

                # Aggiorna il checksum dell'ultima modifica
                self.last_checksums[file_path] = current_checksum

                with lock:
                    for archive_name, archive in list(archives.items()):
                        if file_path in archive['file_paths']:
                            logger.info(f"Modifica intercettata nel file: {file_path}")

                            # Recupera il timestamp attuale del file
                            current_timestamp = os.path.getmtime(file_path)
                            saved_timestamp = archive['modification_timestamps'].get(file_path)
                            logger.info(f"Timestamp corrente del file: {current_timestamp}, Timestamp salvato: {saved_timestamp}")

                            if current_timestamp != saved_timestamp:
                                # Aggiorna il timestamp del file modificato
                                logger.info(f"Timestamp aggiornato per il file: {file_path}")
                                archive['modification_timestamps'][file_path] = current_timestamp
                                with open(archive['timestamps_path'], 'w') as f:
                                    json.dump(archive['modification_timestamps'], f)

                                # Invalidiamo le risposte associate al file modificato
                                invalidate_and_remove_conversation_for_file(archive_name, file_path)

                                # Aggiorna l'indice FAISS per il file modificato
                                index = archive['file_paths'].index(file_path)
                                update_file_in_faiss(archive_name, index, file_path)

                                logger.info(f"Il file '{file_path}' è stato aggiornato nell'indice FAISS e le risposte associate sono state invalidate.")
                            else:
                                logger.info(f"Nessun aggiornamento necessario per il file: {file_path}, i timestamp corrispondono.")
            else:
                logger.info(f"Ignorata modifica per il file: {file_path}, il checksum non è cambiato.")


# Funzione per invalidare le risposte e rimuoverle dal file JSON
def invalidate_and_remove_conversation_for_file(archive_name, file_path):
    """
    Invalida tutte le risposte nella cronologia delle conversazioni associate al file specificato e le rimuove dal file JSON.
    """
    archive = archives[archive_name]
    conversation_path = archive['conversation_history_path']

    if os.path.exists(conversation_path):
        try:
            with open(conversation_path, 'r') as f:
                conversation_history = json.load(f)

            # Nuova lista aggiornata senza le risposte associate al file modificato
            updated_conversation_history = []
            for item in conversation_history:
                if file_path in item['files_used']:
                    logger.info(f"Risposta per la domanda '{item['question']}' invalidata a causa di modifiche nel file '{file_path}'.")
                else:
                    updated_conversation_history.append(item)

            # Salviamo la cronologia aggiornata nel file JSON
            with open(conversation_path, 'w') as f:
                json.dump(updated_conversation_history, f)

            # Aggiorniamo la cronologia anche in memoria
            archive['conversation_history'] = updated_conversation_history
            logger.info(f"Cronologia delle conversazioni aggiornata per l'archivio '{archive_name}' nel file '{conversation_path}'.")

        except Exception as e:
            logger.error(f"Errore durante la rimozione delle risposte dal file '{conversation_path}': {str(e)}")
    else:
        logger.info(f"Il file di cronologia '{conversation_path}' non esiste.")

# Funzione per avviare il monitoraggio dei file per l'archivio corrente
def start_watchdog_for_current_archive():
    """
    Avvia il monitoraggio dei file per l'archivio corrente utilizzando watchdog.
    Se un osservatore è già in esecuzione, lo interrompe e ne avvia uno nuovo.
    """
    logger.info(f"Avvio del monitoraggio per l'archivio '{current_archive_name}'")
    global watchdog_observer
    # Ferma l'osservatore esistente se è già attivo
    if watchdog_observer:
        logger.info("Interruzione dell'osservatore watchdog attivo")
        watchdog_observer.stop()
        watchdog_observer.join()
        watchdog_observer = None

    # Controlla se esiste un archivio e una directory sorgente valida
    archive = archives.get(current_archive_name)
    if archive and archive.get('source_directory'):
        source_directory = archive['source_directory']
        logger.info(f"Directory sorgente trovata: {source_directory}")
        
        monitored_paths = archive.get('monitored_paths', [])

        # Aggiungi i file dalla directory principale ai percorsi monitorati
        logger.info(f"Controllando i file nella directory: {source_directory}")

        for root_dir, _, files in os.walk(source_directory):
            for file in files:
                file_path = os.path.join(root_dir, file)
                if file_path not in monitored_paths:
                    monitored_paths.append(file_path)

        # Verifica se sono stati trovati file da monitorare
        if not monitored_paths:
            logger.warning(f"Nessun file trovato nella directory '{source_directory}'.")
        else:
            logger.info(f"{len(monitored_paths)} file trovati e aggiunti ai percorsi monitorati.")
            for file in monitored_paths:
                logger.info(f"Monitoraggio attivo per il file: {file}")

        # Aggiorna i percorsi monitorati nell'archivio
        archive['monitored_paths'] = monitored_paths

        # Configura l'osservatore watchdog per monitorare le directory
        paths_to_monitor = set(os.path.dirname(path) for path in monitored_paths)
        event_handler = FileChangeHandler()

        # Avvia un nuovo osservatore
        watchdog_observer = Observer()

        # Programma l'osservatore per monitorare le directory
        for path in paths_to_monitor:
            logger.info(f"Monitoraggio avviato per il percorso: {path}")
            watchdog_observer.schedule(event_handler, path=path, recursive=True)

        # Avvia l'osservatore su un thread separato
        watchdog_thread = threading.Thread(target=watchdog_observer.start)
        watchdog_thread.daemon = True
        watchdog_thread.start()

        logger.info(f"Monitoraggio attivo per l'archivio '{current_archive_name}'.")

    else:
        logger.warning("Nessuna directory da monitorare per l'archivio corrente.")

# Funzione per aggiornare un singolo file nell'indice FAISS
def update_file_in_faiss(archive_name, index, file_path):
    with lock:
        try:
            archive = archives[archive_name]
            
            # Ricostruisci l'embedding per il file aggiornato
            logger.info(f"Avvio dell'aggiornamento dell'indice FAISS per il file: {file_path}")
            if '.git' in file_path.split(os.sep):
                content = f"File in directory .git: {os.path.basename(file_path)}"
            elif is_binary_file(file_path):
                content = f"File binario: {os.path.basename(file_path)}"
            else:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                except Exception as e:
                    content = f"Errore nella lettura del file: {str(e)}"
                    logger.error(f"Errore nella lettura del file '{file_path}': {str(e)}")
                    return

            # Log del contenuto per debug
            logger.info(f"Contenuto del file '{file_path}' recuperato con successo.")

            # Crea il nuovo embedding del file aggiornato
            new_embedding = embeddings.embed_documents([content])[0]
            logger.info(f"Nuovo embedding generato per il file '{file_path}'.")

            # Sostituisci il vettore esistente nell'indice FAISS con il nuovo embedding
            if archive['faiss_index'] is not None:
                logger.info(f"Avvio della sostituzione dell'embedding nell'indice FAISS per il file '{file_path}'.")
                archive['faiss_index'].replace_vectors(np.array([new_embedding]), [index])
                logger.info(f"Indice FAISS aggiornato per il file '{file_path}' con successo.")

            # Log specifico per indicare il completamento della sostituzione
            logger.info(f"Embedding per il file '{file_path}' estrapolato e sostituito con successo nell'indice FAISS.")

            # Aggiorna il docstore e il mapping dell'indice
            update_index_to_docstore(archive_name)
            logger.info(f"Docstore e mapping aggiornati per il file '{file_path}'.")

        except Exception as e:
            logger.error(f"Errore durante l'aggiornamento del file '{file_path}': {str(e)}")


# Funzione per inviare il prompt a ChatGPT con i documenti pertinenti
def query_chatgpt_with_documents(prompt):
    """
    Esegue una query a ChatGPT utilizzando i documenti pertinenti dall'archivio corrente.
    Prima di inviare la query, verifica se la domanda è già presente nella cronologia delle conversazioni
    e se i file associati alla risposta non sono stati modificati.
    """
    global current_archive_name
    if current_archive_name is None:
        messagebox.showerror("Errore", "Per favore, crea o seleziona un archivio prima!")
        return

    archive = archives[current_archive_name]

    # Controlla se la domanda è già stata posta nella cronologia
    normalized_prompt = prompt.strip().lower()  # Normalizziamo il prompt
    for item in archive['conversation_history']:
        if item['question'].strip().lower() == normalized_prompt:
            # Controlla se i file utilizzati per generare questa risposta sono stati modificati
            if all(os.path.exists(f) and os.path.getmtime(f) == item['file_timestamps'].get(f, 0) for f in item['files_used']):
                logger.info(f"Domanda trovata nella cronologia e i file non sono stati modificati. Risposta recuperata.")
                return item['answer']  # Restituisce direttamente la risposta dalla cronologia
            else:
                # Se i file sono stati modificati, elimina la risposta obsoleta
                logger.info(f"File modificati per la domanda '{prompt}'. Risposta invalidata e rimossa.")
                archive['conversation_history'].remove(item)
                save_conversation_history()  # Salva la cronologia aggiornata
                break

    # Se la domanda non è nella cronologia o i file sono stati modificati, esegui la query su ChatGPT
    if archive['faiss_index'] is None or not archive['file_paths']:
        messagebox.showerror("Errore", "L'indice FAISS o i percorsi dei file non sono stati caricati per l'archivio corrente!")
        return

    # Crea il FAISS vector store con InMemoryDocstore
    vectorstore = FAISS(embedding_function=embeddings.embed_query, index=archive['faiss_index'],
                        docstore=archive['docstore'], index_to_docstore_id=archive['index_to_docstore_id'])

    # Recupera i documenti rilevanti
    docs = vectorstore.similarity_search(prompt, k=5)
    files_used = [doc.metadata['source'] for doc in docs]

    # Recupera i timestamp dei file
    file_timestamps = {file: os.path.getmtime(file) for file in files_used}

    # Crea la memoria della conversazione
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for item in archive['conversation_history']:
        memory.chat_memory.add_user_message(item['question'])
        memory.chat_memory.add_ai_message(item['answer'])

    # Crea la catena di conversazione con recupero
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        verbose=True,
        max_tokens_limit=max_tokens_var.get()  # Imposta il limite di token
    )

    # Esegui la query su ChatGPT con i documenti pertinenti
    try:
        response = qa_chain({"question": prompt})
        answer = response['answer']

        # Aggiungi la nuova domanda, risposta, file utilizzati e timestamp alla cronologia e salvala
        archive['conversation_history'].append({
            'question': prompt,
            'answer': answer,
            'files_used': files_used,
            'file_timestamps': file_timestamps
        })
        save_conversation_history()

        return answer
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione della query: {str(e)}")
        messagebox.showerror("Errore", f"Errore durante l'esecuzione della query: {str(e)}")

# Funzione per inviare i file selezionati a ChatGPT
def query_chatgpt_with_selected_files(file_paths):
    """
    Esegue una query a ChatGPT utilizzando i file selezionati dall'utente.
    """
    global current_archive_name
    archive = archives[current_archive_name]

    # Leggi i contenuti dei file selezionati
    docs = []
    for file_path in file_paths:
        if '.git' in file_path.split(os.sep):
            content = f"File in directory .git: {os.path.basename(file_path)}"
        elif is_binary_file(file_path):
            content = f"File binario: {os.path.basename(file_path)}"
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                content = f"Errore nella lettura del file: {str(e)}"
                logger.error(f"Errore nella lettura del file '{file_path}': {str(e)}")

        doc = Document(page_content=content, metadata={"source": file_path})
        docs.append(doc)

    # Crea la memoria della conversazione
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Carica la cronologia delle conversazioni
    for item in archive['conversation_history']:
        memory.chat_memory.add_user_message(item['question'])
        memory.chat_memory.add_ai_message(item['answer'])

    # Crea la catena di conversazione
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=None,  # Non utilizziamo il retriever in questo caso
        memory=memory,
        verbose=True,
        max_tokens_limit=max_tokens_var.get()  # Imposta il limite di token
    )

    # Esegui la query su ChatGPT con i documenti selezionati
    try:
        prompt = prompt_entry.get("1.0", tk.END).strip()
        response = qa_chain({"question": prompt, "input_documents": docs})
        answer = response['answer']

        # Aggiorna la cronologia delle conversazioni
        archive['conversation_history'].append({
            'question': prompt,
            'answer': answer,
            'files_used': [doc.metadata['source'] for doc in docs],
            'file_timestamps': {file: os.path.getmtime(file) for file in [doc.metadata['source'] for doc in docs]}
        })
        save_conversation_history()

        return answer
    except Exception as e:
        if 'context_length_exceeded' in str(e):
            messagebox.showerror("Errore", "La query ha superato il limite massimo di contesto del modello. Riduci la quantità di contenuto.")
        else:
            logger.error(f"Errore durante l'esecuzione della query: {str(e)}")
            messagebox.showerror("Errore", f"Errore durante l'esecuzione della query: {str(e)}")

# Funzione per gestire l'invio del prompt
def send_prompt():
    """
    Gestisce l'invio del prompt inserito dall'utente e visualizza la risposta.
    """
    prompt = prompt_entry.get("1.0", tk.END).strip()
    if prompt:
        result = query_chatgpt_with_documents(prompt)
        if result:
            format_chatgpt_response(result)
    else:
        messagebox.showerror("Errore", "Per favore, inserisci un prompt prima di inviare!")

# Funzione per formattare la risposta di ChatGPT
def format_chatgpt_response(response):
    """
    Formatta la risposta di ChatGPT per visualizzare i blocchi di codice con uno sfondo diverso e aggiunge un pulsante per copiare il codice.
    """
    result_text.configure(state='normal')
    result_text.delete("1.0", tk.END)

    # Separiamo la risposta in righe e formattiamo il codice
    lines = response.split("\n")
    in_code_block = False
    code_content = ""
    for line in lines:
        if line.strip().startswith("```"):
            if not in_code_block:
                # Inizia un blocco di codice
                in_code_block = True
                code_content = ""
            else:
                # Termina un blocco di codice
                in_code_block = False
                # Mostra il codice con sfondo diverso e pulsante copia
                display_code_block(code_content)
        elif in_code_block:
            code_content += line + "\n"
        else:
            result_text.insert(tk.END, line + "\n")
    result_text.configure(state='disabled')

# Funzione per visualizzare un blocco di codice con sfondo diverso e pulsante copia
def display_code_block(code_content):
    """
    Visualizza un blocco di codice con sfondo grigio e aggiunge un pulsante per copiare il codice.
    """
    code_frame = tk.Frame(result_text, bg="#f0f0f0")
    code_frame.pack(fill="x", padx=5, pady=5)

    code_text = tk.Text(code_frame, height=10, wrap="none", bg="#f0f0f0", fg="#000000")
    code_text.insert(tk.END, code_content)
    code_text.configure(state='disabled')
    code_text.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(code_frame, command=code_text.yview)
    scrollbar.pack(side="right", fill="y")
    code_text['yscrollcommand'] = scrollbar.set

    copy_button = tk.Button(code_frame, text="Copia Codice", command=lambda: copy_code(code_content))
    copy_button.pack(side="bottom", pady=5)

# Funzione per copiare il codice negli appunti
def copy_code(code_content):
    """
    Copia il contenuto del codice negli appunti.
    """
    root.clipboard_clear()
    root.clipboard_append(code_content)
    messagebox.showinfo("Copiato", "Il codice è stato copiato negli appunti!")

# Funzione per gestire la selezione di più file
def select_multiple_files():
    """
    Apre una finestra di dialogo per selezionare più file e li indica nell'archivio corrente.
    """
    file_paths = filedialog.askopenfilenames(title="Seleziona i file da indicizzare")
    if file_paths:
        index_multiple_files(file_paths)

# Funzione per gestire la selezione di più directory tramite una finestra di dialogo personalizzata
def select_multiple_directories():
    """
    Permette la selezione di più directory contemporaneamente tramite una finestra di dialogo personalizzata.
    """
    selected_directories = []

    def add_directory():
        directory = filedialog.askdirectory(title="Seleziona una cartella")
        if directory:
            directories_listbox.insert(tk.END, directory)

    def remove_directory():
        selected_items = directories_listbox.curselection()
        for index in reversed(selected_items):
            directories_listbox.delete(index)

    def start_indexing():
        directories = directories_listbox.get(0, tk.END)
        if directories:
            for directory in directories:
                index_directory(directory)
            dir_window.destroy()
        else:
            messagebox.showwarning("Attenzione", "Nessuna directory selezionata.")

    # Funzione per indicizzare una directory
    def index_directory(directory):
        file_paths = []
        for root_dir, _, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root_dir, file))
        if file_paths:
            index_multiple_files(file_paths)

    # Crea una nuova finestra
    dir_window = tk.Toplevel(root)
    dir_window.title("Seleziona Directory")

    # Lista delle directory selezionate
    directories_listbox = tk.Listbox(dir_window, selectmode=tk.MULTIPLE, width=80, height=15)
    directories_listbox.pack(padx=10, pady=10)

    # Pulsanti per aggiungere o rimuovere directory
    buttons_frame = tk.Frame(dir_window)
    buttons_frame.pack()

    add_button = tk.Button(buttons_frame, text="Aggiungi Directory", command=add_directory)
    add_button.pack(side=tk.LEFT, padx=5, pady=5)

    remove_button = tk.Button(buttons_frame, text="Rimuovi Directory", command=remove_directory)
    remove_button.pack(side=tk.LEFT, padx=5, pady=5)

    start_button = tk.Button(dir_window, text="Inizia Indicizzazione", command=start_indexing)
    start_button.pack(pady=10)

# Funzione per aprire la finestra dei log
def open_log_window():
    """
    Apre una finestra che mostra i log dell'applicazione.
    """
    log_window = tk.Toplevel(root)
    log_window.title("Console dei Log")
    log_text = tk.Text(log_window, state='disabled', width=80, height=20)
    log_text.pack(fill='both', expand=True)
    # Configura il logger per scrivere nella Text widget
    global log_handler
    log_handler = TextHandler(log_text)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)

# Classe per gestire i log nella Text widget
class TextHandler(logging.Handler):
    """
    Custom logging handler che scrive i log in una Text widget di Tkinter.
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            # Scorri automaticamente verso il basso
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)

# Funzione per selezionare i file da inviare a ChatGPT
def select_files_for_chatgpt():
    """
    Permette all'utente di selezionare uno o più file associati all'archivio da inviare a ChatGPT.
    """
    global current_archive_name
    if current_archive_name is None:
        messagebox.showerror("Errore", "Per favore, seleziona un archivio prima!")
        return

    archive = archives[current_archive_name]
    if not archive['file_paths']:
        messagebox.showerror("Errore", "Nessun file è stato indicizzato per l'archivio selezionato.")
        return

    # Permetti la selezione multipla di file dall'archivio corrente
    selected_files = filedialog.askopenfilenames(title="Seleziona file da inviare a ChatGPT",
                                                 initialdir=os.path.dirname(archive['file_paths'][0]),
                                                 filetypes=[("Tutti i file", "*.*")])

    # Verifica che i file selezionati siano nell'archivio
    selected_files = [f for f in selected_files if f in archive['file_paths']]

    if not selected_files:
        messagebox.showerror("Errore", "I file selezionati non sono presenti nell'archivio corrente.")
        return

    # Esegui la query a ChatGPT con i file selezionati
    result = query_chatgpt_with_selected_files(selected_files)
    if result:
        format_chatgpt_response(result)

# Sezione per l'indicizzazione
index_frame = tk.LabelFrame(main_frame, text="Indicizzazione")
index_frame.pack(fill="x", pady=5)

# Pulsante per caricare e indicizzare più file
button_file = tk.Button(index_frame, text="Carica e Indicizza File", command=select_multiple_files)
button_file.pack(side="left", padx=5, pady=5)

# Pulsante per caricare e indicizzare più directory
button_dir = tk.Button(index_frame, text="Carica e Indicizza Directory", command=select_multiple_directories)
button_dir.pack(side="left", padx=5, pady=5)

# Pulsante per salvare l'indice FAISS
button_save = tk.Button(index_frame, text="Salva Indice FAISS", command=save_faiss_index)
button_save.pack(side="left", padx=5, pady=5)

# Pulsante per aprire la console dei log
button_open_log = tk.Button(index_frame, text="Apri Console dei Log", command=open_log_window)
button_open_log.pack(side="left", padx=5, pady=5)

# Sezione per l'interazione con ChatGPT
chat_frame = tk.LabelFrame(main_frame, text="Interazione con ChatGPT")
chat_frame.pack(fill="x", pady=5)

# Campo di input per il prompt (aumentato in dimensione)
prompt_label = tk.Label(chat_frame, text="Inserisci il prompt per ChatGPT:")
prompt_label.pack(anchor="w", padx=5, pady=5)

prompt_entry = tk.Text(chat_frame, width=80, height=5)
prompt_entry.pack(fill="x", padx=5, pady=5)
prompt_entry.focus_set()

# Pulsante per inviare il prompt a ChatGPT
button_send_prompt = tk.Button(chat_frame, text="Invia Prompt", command=send_prompt)
button_send_prompt.pack(pady=5)

# Pulsante per selezionare file da inviare a ChatGPT
button_select_files = tk.Button(chat_frame, text="Seleziona File per ChatGPT", command=select_files_for_chatgpt)
button_select_files.pack(pady=5)

# Campo di testo per visualizzare la risposta
result_label = tk.Label(chat_frame, text="Risposta di ChatGPT:")
result_label.pack(anchor="w", padx=5, pady=5)

result_text = tk.Text(chat_frame, height=15, state='disabled')
result_text.pack(fill="both", padx=5, pady=5)

# Carica tutti gli archivi esistenti all'avvio
load_all_archives()

# Funzione per fermare il watchdog all'uscita o durante il cambio di archivio
def stop_watchdog():
    global watchdog_observer
    if watchdog_observer:
        logger.info("Fermando l'osservatore watchdog in esecuzione")
        watchdog_observer.stop()
        watchdog_observer.join()
        watchdog_observer = None

# Gestione della chiusura della finestra
def on_closing():
    stop_watchdog()  # Ferma il watchdog quando l'applicazione viene chiusa
    root.destroy()

# Associa la funzione di chiusura all'evento di uscita
root.protocol("WM_DELETE_WINDOW", on_closing)

# Avvio dell'interfaccia grafica
root.mainloop()
