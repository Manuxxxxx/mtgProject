import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import os
import random
import requests
from io import BytesIO

# === CONFIG ===
SETS_TO_INCLUDE = ['tstx', 'tinr', 'mh3', 'prm', 'ulg', 'tddi', 'dis', '7ed', 'mm2', 'eve', 'ttdc', 'dst', 's99', 'pf24', 'scg', 'tfic', 'te01', 'exo', 'ysnc', 'wc02', 'ima', 'tblc', 'mznr', 'tc20', 'tneo', 'tdmu', 'tfth', 'm3c', 'ybro', 'iko', 'pca', 'ths', 'tmoc', 'dsc', 'oana', 'wc04', 'cc2', 'onc', 'ydsk', 'tnec', 'opca', 'bng', 'mom', 'tlci', 'ttsr', 'ltc', 'w17', 'fj25', 'tdmc', 'troe', 'jud', 'woc', 'ainr', 'ncc', 'tcm2', 'tbot', 'leg', 'ktk', 'nem', '9ed', 'mneo', 'dmc', 'tor', 'pmic', 'trna', 'apc', 'pmoa', 'wc99', 'twar', 'ust', 'amh3', 'ywoe', 'oarc', 'cma', 'tala', 'tdmr', 'unk', 'akh', 'aer', 'tddt', 'totj', 'aotj', 'otc', 'aafr', 'tmid', 'thp3', 'rex', 'grn', 'phel', 'amkm', 'arc', 'tuma', 'afc', '2xm', 'atdm', 'jou', 'tsb', 'afr', 'ptg', 'por', 'pip', 'tdvd', 'tvow', 'clu', 'td2', 'cmm', 'mir', 'stx', 'thou', 'fclu', 'lgn', 'fdmu', 'ice', 'anb', 'ydmu', 'drc', 'dgm', '2ed', 'und', 'zen', 'ddq', 'tcns', 'tbfz', 'tund', 'neo', 'avr', 'dft', 'ddh', 'tltr', 'tznr', 'tc19', '5dn', 'admu', '6ed', 'tdc', 'ddk', 'mkm', 'brc', 'nph', 'f12', 'mor', 'dka', 'evg', 'tzen', 'twho', 'fbro', 'wc03', 'tmh2', 'ddn', 'soi', 'moc', 'thb', 'tbrc', 'bro', 'me3', 'bfz', 'm14', 't2x2', 'tmom', 'l17', 'khm', 'f18', 'temn', 'drk', 'tbng', 'phtr', 'hml', 'tm3c', 'tdka', 'one', 'ddf', 'm11', 'ody', 'bot', 'gpt', 'altr', 'tdag', 'tsnc', 'tkhm', 'blc', 'ablb', 'mvow', 'dom', 'fj22', 'tc15', 'avow', 'bok', 't10e', 'cns', 'ph19', 'tpr', 'dmu', 'mh1', 'xln', 'm15', 'tbbd', '8ed', 'hop', 'ddo', 'aclb', '40k', 'tcmm', 'tmkm', 'rix', 'mma', 'gn3', 'hou', 'tmh1', 'eld', 'ddt', 'slx', 'fut', 'hbg', 'mbs', 'ddp', 'mrd', 'tm20', 'tgrn', 'tdsk', 'znr', 'twoe', 'ymkm', 'togw', 'ddl', 'wwk', 'thp1', 'voc', 'ddr', 'tarb', 'mafr', 'tdtk', 'tltc', 'cmr', 'tclb', 'tmh3', 'amid', 'm12', 'alci', 'ddu', 'aacr', 'clb', 'tust', 'emn', 'tkhc', 'ons', 'tblb', 'rtr', 'hho', 'tsoi', 'tiko', 'wc01', 'pz2', 'tths', 'dci', 'dtk', 'chk', 'tbig', 'arn', 'rvr', 'tbth', 'dvd', 'pemn', 'yotj', 'uma', 'tonc', 'fltr', 'adft', 'atq', 'p02', 'cc1', 'tsr', 'a25', 'c18', 'tone', 'c13', 'tdft', 'rna', 'ydft', 'roe', 'cm2', 'dds', 'twoc', 'frf', 'tkld', 'dmr', 'tori', 'rav', 'taer', 'akhm', 'kld', 'c17', 'ala', 'tc14', 'vma', 'ptbro', 'khc', 'w16', 'f17', 'h2r', 'som', 'amh2', 'ddj', 'wth', 'tm21', 'ema', 'ph17', 'ptk', 'tacr', 'ytdm', 'me2', 'csp', 'abro', 'fin', 'mm3', 'twwk', '2x2', 'snc', 'e01', 'mmq', 'sunf', 'pcy', 'j21', 'tc17', 'tthb', 'ddm', 'tmkc', 'totp', 'm19', 'pvan', 'tunf', 'lcc', 'tpip', 'ori', '5ed', 'plst', 'tafr', 'tm15', 'gtc', 'blb', 'ph21', 'arb', 'con', 'unf', 'cmd', 'p03', 'ttdm', 'spe', 'tgk2', 'who', 'q07', 'yneo', 'mat', 'c15', 'ddi', 'me4', 'psdg', 't40k', 'lci', 'tsom', 'ppc1', 'fone', 'sth', 'tpca', 'macr', 'j25', 'tc21', 'tshm', 'aneo', 'asnc', 'totc', 'vis', 'c14', '10e', 'teld', 'ffdn', 'tm14', 'tm19', 'tdrc', 'all', 'mid', '4ed', 'ptdmu', 'znc', 'tdde', 'm13', 'j22', 'tgk1', 'pls', 'woe', 'tdsc', 'trtr', 'mone', 'tdom', 'ymid', 'shm', 'cn2', 'tgtc', 'nec', 'aone', 'mkhm', 'pf25', 'inv', 'oe01', 'ugl', 'fem', 'chr', 'm21', 'tktk', 'tvoc', 'c21', 'tcn2', 'tcma', 'me1', 'fmom', 'l16', 'isd', 'mbro', 'c20', 'war', 'aznr', 'afdn', 'tafc', 'usg', 'c19', 'tmp', 'adsk', 'tsp', 'altc', 'fic', 'lrw', 'tfin', 'acmm', 'dde', 'mb2', 'mdmu', 'amom', 'wc98', 'ylci', 'txln', 'mmid', 'uds', 'sok', 'ta25', 'mstx', 'gn2', 'mmh2', 'tisd', 'dsk', 'mclb', 'ltr', 'pcel', 'tlrw', 'm20', 'trix', 'h17', 'msnc', 'sld', 'wc00', 'jmp', 'unh', 'tima', 'gs1', 'mic', 'jvc', 'bbd', 'ph22', 'amh1', 'm10', 'ddg', 'tmbs', 'smh3', 'tfdn', 'acr', 'tdm', 'mh2', 'tmed', 'fdn', 'big', 'yblb', 'wc97', 'tc18', 'tcmr', 'ph18', 'ph20', 'otj', 't2xm', 'tm11', 'tjou', 'cmb2', 'mkc', 'ogw', 'astx', 'gvl', 'thp2', 'gnt', 'tugl', 'trvr', 'past', 'trex', 'fjmp', 's00', 'yone', 'plc', 'sum', 'takh', 'vow', 'inr', 'tbro', 'awoe', 'tmm3', 'c16', 'tema', 'tmma', 'pssc']
BULK_FILE = 'datasets/processed/tag_included/cards_with_tags_20250622170831_withuri.json'
SYNERGY_FILE = 'edhrec_data/labeled/with_random/random_real_synergies_copy.json'
IMAGE_CACHE_DIR = 'image-dataset/'

def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_or_download_image(card):
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    name = card["name"].replace("/", "_").replace(" ", "_")
    image_path = os.path.join(IMAGE_CACHE_DIR, f"{name}.png")

    if os.path.exists(image_path):
        img = Image.open(image_path)
    else:
        url = card.get("image_uris", {}).get("png")
        if not url:
            return None
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(image_path)
    return img

def resize_image(img, height=400):
    w, h = img.size
    new_w = int((height / h) * w)
    return img.resize((new_w, height), Image.LANCZOS)

# === MAIN APP ===
class SynergyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MTG Synergy Labeler")
        self.root.configure(bg="#f0f0f0")

        self.cards = [c for c in load_json(BULK_FILE) if c["set"] in SETS_TO_INCLUDE]
        self.synergies = load_json(SYNERGY_FILE) if os.path.exists(SYNERGY_FILE) else []
        self.synergy_map = self.build_synergy_map(self.synergies)
        self.history = []
        self.current_pair = None

        self.setup_ui()
        self.load_new_pair()

    def build_synergy_map(self, synergies):
        mapping = {}
        for s in synergies:
            key = self.pair_key(s["card1"]["name"], s["card2"]["name"])
            mapping[key] = s
        return mapping

    def pair_key(self, name1, name2):
        return "".join(sorted([name1, name2]))

    def setup_ui(self):
        self.card_frames = []
        self.image_labels = []
        self.text_boxes = []
        self.search_boxes = []
        self.text_vars = []

        top_frame = tk.Frame(self.root, bg="#f0f0f0")
        top_frame.pack(fill="both", expand=True)

        for side in [0, 1]:
            frame = tk.Frame(top_frame, bg="#f0f0f0")
            frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

            img_label = tk.Label(frame, bg="#f0f0f0")
            img_label.pack()

            text = tk.Text(frame, height=15, width=80, wrap="word", font=("Arial", 24), state="disabled")
            text.pack()

            var = tk.StringVar()
            search = ttk.Combobox(frame, textvariable=var, font=("Arial", 12))
            search.pack(pady=5, fill='x')
            search.bind("<KeyRelease>", lambda e, s=side: self.update_suggestions(e, s))
            search.bind("<<ComboboxSelected>>", lambda e, s=side: self.replace_card(e, s))

            self.card_frames.append(frame)
            self.image_labels.append(img_label)
            self.text_boxes.append(text)
            self.search_boxes.append(search)
            self.text_vars.append(var)

        self.info_label = tk.Label(self.root, text="", fg="blue", font=("Arial", 14), bg="#f0f0f0")
        self.info_label.pack(pady=5)

        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=10)

        self.back_btn = tk.Button(button_frame, text="Back", command=self.go_back, width=30, height=4, font=("Arial", 18))
        self.back_btn.pack(side="left", padx=5)

        self.yes_btn = tk.Button(button_frame, text="SYNERGY", command=lambda: self.label_synergy(1), width=30, height=4, font=("Arial", 18), bg="#d1ffd1")
        self.yes_btn.pack(side="left", padx=5)

        self.no_btn = tk.Button(button_frame, text="NO SYNERGY", command=lambda: self.label_synergy(0), width=30, height=4, font=("Arial", 18), bg="#ffd1d1")
        self.no_btn.pack(side="left", padx=5)

        self.next_btn = tk.Button(button_frame, text="NEXT", command=self.load_new_pair, state="disabled", width=30, height=4, font=("Arial", 18))
        self.next_btn.pack(side="left", padx=5)

        self.status_label = tk.Label(self.root, text="", font=("Arial", 32, "bold"), fg="red", bg="#f0f0f0")
        self.status_label.pack(pady=5)

    def display_cards(self):
        for i, card in enumerate(self.current_pair):
            img = load_or_download_image(card)
            if img:
                img = resize_image(img)
                tk_img = ImageTk.PhotoImage(img)
                self.image_labels[i].configure(image=tk_img)
                self.image_labels[i].image = tk_img

            text = f"Name: {card['name']}\nType: {card['type_line']}\nText: {card.get('oracle_text', '')}"
            if "power" in card:
                text += f"\nP/T: {card['power']} / {card['toughness']}"
            self.text_boxes[i].config(state="normal")
            self.text_boxes[i].delete("1.0", tk.END)
            self.text_boxes[i].insert(tk.END, text)
            self.text_boxes[i].config(state="disabled")
            self.text_vars[i].set(card["name"])

        key = self.pair_key(self.current_pair[0]['name'], self.current_pair[1]['name'])
        if key in self.synergy_map:
            existing = self.synergy_map[key]
            tags = []
            if "synergy_edhrec" in existing:
                tags.append("edhrec")
            elif "synergy_syntethic" in existing:
                tags.append("synthetic")
            if "synergy_manual" in existing:
                tags.append("manual")
                self.status_label.config(text=f"SYNERGY = {existing['synergy_manual']}")
                self.yes_btn.config(state="disabled")
                self.no_btn.config(state="disabled")
                self.next_btn.config(state="normal")
            else:
                self.status_label.config(text="")
                self.yes_btn.config(state="normal")
                self.no_btn.config(state="normal")
                self.next_btn.config(state="disabled")
            self.info_label.config(text=" / ".join(tags))
        else:
            self.info_label.config(text="")
            self.status_label.config(text="")
            self.yes_btn.config(state="normal")
            self.no_btn.config(state="normal")
            self.next_btn.config(state="disabled")

    def update_suggestions(self, event, index):
        typed = self.text_vars[index].get().lower()
        suggestions = [c["name"] for c in self.cards if typed in c["name"].lower()][:10]
        self.search_boxes[index]["values"] = suggestions

    def replace_card(self, event, index):
        name = self.text_vars[index].get()
        matches = [c for c in self.cards if c["name"] == name]
        if matches:
            self.current_pair[index] = matches[0]
            self.display_cards()

    def label_synergy(self, value):
        card1, card2 = self.current_pair
        key = self.pair_key(card1["name"], card2["name"])
        entry = self.synergy_map.get(key, {
            "card1": {"name": card1["name"]},
            "card2": {"name": card2["name"]},
        })
        entry["synergy_manual"] = value
        self.synergy_map[key] = entry
        self.synergies = list(self.synergy_map.values())
        save_json(self.synergies, SYNERGY_FILE)
        self.status_label.config(text=f"SYNERGY = {value}")
        self.next_btn.config(state="normal")
        self.yes_btn.config(state="disabled")
        self.no_btn.config(state="disabled")

    def load_new_pair(self):
        attempts = 0
        while attempts < 100:
            card1, card2 = random.sample(self.cards, 2)
            if card1["name"] == card2["name"]:
                continue
            key = self.pair_key(card1["name"], card2["name"])
            if key not in self.synergy_map:
                break
            attempts += 1
        self.current_pair = [card1, card2]
        self.display_cards()

    def go_back(self):
        previous = [s for s in reversed(self.synergies) if "synergy_manual" in s]
        if previous:
            last = previous[0]
            c1 = next((c for c in self.cards if c["name"] == last["card1"]["name"]), None)
            c2 = next((c for c in self.cards if c["name"] == last["card2"]["name"]), None)
            if c1 and c2:
                self.current_pair = [c1, c2]
                self.display_cards()

if __name__ == "__main__":
    root = tk.Tk()
    app = SynergyApp(root)
    root.mainloop()


def downlaod_all_images():
    """
    Download all images from the bulk file and save them in the IMAGE_CACHE_DIR.
    """
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    cards = load_json(BULK_FILE)

    for card in cards:
        name = card["name"].replace("/", "_").replace(" ", "_")
        image_path = os.path.join(IMAGE_CACHE_DIR, f"{name}.png")

        if not os.path.exists(image_path):
            url = card.get("image_uris", {}).get("png")
            if not url:
                continue
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save(image_path)
            print(f"Downloaded {name}")
        # else:
        #     print(f"Image for {name} already exists.")