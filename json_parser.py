import json
import time
import conf

mtg_keywords = {
    "Deathtouch": "Any amount of damage this deals to a creature is enough to destroy it.",
    "Defender": "This creature can't attack.",
    "Double Strike": "This creature deals both first-strike and regular combat damage.",
    "Enchant": "This Aura can only be attached to a specified type of permanent.",
    "Equip": "Attach this Equipment to a creature you control by paying the equip cost.",
    "First Strike": "This creature deals combat damage before creatures without first strike.",
    "Flash": "You may cast this spell any time you could cast an instant.",
    "Flying": "This creature can't be blocked except by creatures with flying or reach.",
    "Haste": "This creature can attack and {T} as soon as it comes under your control.",
    "Hexproof": "This permanent can't be the target of spells or abilities your opponents control.",
    "Indestructible": "Effects that say 'destroy' don't destroy this creature. It can't be destroyed by damage.",
    "Lifelink": "Damage dealt by this creature also causes you to gain that much life.",
    "Menace": "This creature can't be blocked except by two or more creatures.",
    "Prowess": "Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.",
    "Reach": "This creature can block creatures with flying.",
    "Trample": "Excess damage this creature deals to blockers is dealt to the defending player or planeswalker.",
    "Vigilance": "Attacking doesn't cause this creature to tap.",
    "Shroud": "This permanent can't be the target of any spells or abilities.",
    "Protection": "This permanent can't be targeted, dealt damage, enchanted, or blocked by anything of the specified quality.",
    "Flashback": "You may cast this card from your graveyard for its flashback cost. Then exile it.",
    "Cycling": "Pay the cycling cost and discard this card to draw a card.",
    "Morph": "You may cast this card face down as a 2/2 creature for {3}. Turn it face up any time for its morph cost.",
    "Persist": "When this creature dies, if it had no -1/-1 counters on it, return it to the battlefield under its owner's control with a -1/-1 counter on it.",
    "Undying": "When this creature dies, if it had no +1/+1 counters on it, return it to the battlefield under its owner's control with a +1/+1 counter on it.",
    "Scry": "Look at the top card(s) of your library, then put any number of them on the bottom of your library and the rest on top in any order.",
    "Cascade": "When you cast this spell, exile cards from the top of your library until you exile a nonland card that costs less. You may cast it without paying its mana cost.",
    "Convoke": "Your creatures can help cast this spell. Each creature you tap while casting this spell pays for {1} or one mana of that creature's color.",
    "Delve": "Each card you exile from your graveyard while casting this spell pays for {1}.",
    "Evoke": "You may cast this spell for its evoke cost. If you do, it's sacrificed when it enters the battlefield.",
    "Kicker": "You may pay an additional cost as you cast this spell to gain an additional effect.",
    "Landwalk": "This creature can't be blocked as long as defending player controls a land of the specified type.",
    "Modular": "This creature enters the battlefield with a number of +1/+1 counters on it. When it dies, you may put its +1/+1 counters on target artifact creature.",
    "Ninjutsu": "Pay the ninjutsu cost and return an unblocked attacker you control to hand: Put this card onto the battlefield from your hand tapped and attacking.",
    "Overload": "You may cast this spell for its overload cost. If you do, change its text by replacing all instances of 'target' with 'each.'",
    "Phasing": "This permanent phases in and out of play, meaning it alternates between being on the battlefield and being treated as though it doesn't exist.",
    "Provoke": "When this creature attacks, you may have target creature defending player controls untap and block it if able.",
    "Rampage": "Whenever this creature becomes blocked, it gets +X/+X until end of turn for each creature blocking it beyond the first.",
    "Rebound": "If you cast this spell from your hand, exile it as it resolves. At the beginning of your next upkeep, you may cast it from exile without paying its mana cost.",
    "Retrace": "You may cast this card from your graveyard by discarding a land card in addition to paying its other costs.",
    "Scavenge": "Exile this card from your graveyard: Put a number of +1/+1 counters equal to this card's power on target creature. Scavenge only as a sorcery.",
    "Split Second": "As long as this spell is on the stack, players can't cast spells or activate abilities that aren't mana abilities.",
    "Storm": "When you cast this spell, copy it for each spell cast before it this turn. You may choose new targets for the copies.",
    "Suspend": "Rather than cast this card from your hand, you may pay its suspend cost and exile it with a number of time counters on it. At the beginning of your upkeep, remove a time counter. When the last is removed, cast it without paying its mana cost.",
    "Transmute": "Pay the transmute cost and discard this card: Search your library for a card with the same converted mana cost, reveal it, and put it into your hand. Then shuffle your library. Transmute only as a sorcery.",
    "Unearth": "Return this card from your graveyard to the battlefield. It gains haste. Exile it at the beginning of the next end step or if it would leave the battlefield. Unearth only as a sorcery.",
    "Wither": "This deals damage to creatures in the form of -1/-1 counters.",
    "Affinity": "This spell costs {1} less to cast for each permanent you control of the specified type.",
    "Annihilator": "Whenever this creature attacks, defending player sacrifices a number of permanents equal to the annihilator number.",
    "Bestow": "If you cast this card for its bestow cost, it's an Aura spell with enchant creature. It becomes a creature again if it's not attached to a creature.",
    "Bloodthirst": "If an opponent was dealt damage this turn, this creature enters the battlefield with additional +1/+1 counters.",
    "Bushido": "When this blocks or becomes blocked, it gets +X/+X until end of turn.",
    "Champion": "When this enters the battlefield, sacrifice it unless you exile another creature you control of the specified type. When this leaves the battlefield, that card returns to the battlefield.",
    "Cipher": "Then you may exile this spell card encoded on a creature you control. Whenever that creature deals combat damage to a player, its controller may cast a copy of the encoded card without paying its mana cost.",
    "Dredge": "If you would draw a card, you may instead put a number of cards from the top of your library into your graveyard and return this card from your graveyard to your hand.",
    "Entwine": "You may choose both modes of a spell with entwine if you pay the entwine cost.",
    "Epic": "For the rest of the game, you can't cast spells. At the beginning of each of your upkeeps, copy this spell except for its epic ability.",
    "Exalted": "Whenever a creature you control attacks alone, that creature gets +1/+1 until end of turn.",
    "Exploit": "When this creature enters the battlefield, you may sacrifice a creature. If you do, it gains an additional effect.",
    "Extort": "Whenever you cast a spell, you may pay {W/B}. If you do, each opponent loses 1 life and you gain that much life.",
    "Fading": "This permanent enters the battlefield with a number of fade counters. At the beginning of your upkeep, remove a fade counter. If you can't, sacrifice it.",
    "Fear": "This creature can't be blocked except by artifact creatures and/or black creatures.",
    "Forecast": "During your upkeep, you may reveal this card from your hand and pay its forecast cost. If you do, an effect occurs.",
    "Graft": "This creature enters the battlefield with a number of +1/+1 counters. Whenever another creature enters the battlefield, you may move a +1/+1 counter from this creature onto it.",
    "Haunt": "When this card is put into a graveyard, exile it haunting target creature. When that creature dies, you may apply an effect.",
    "Hideaway": "This land enters the battlefield tapped. When it does, look at the top four cards of your library, exile one face down, then put the rest on the bottom of your library. You may play the exiled card without paying its mana cost if a certain condition is met.",
    "Infect": "This creature deals damage to creatures in the form of -1/-1 counters and to players in the form of poison counters.",
    "Intimidate": "This creature can't be blocked except by artifact creatures and/or creatures that share a color with it.",
    "Madness": "If you discard this card, you may cast it for its madness cost instead of putting it into your graveyard.",
    "Miracle": "You may cast this card for its miracle cost when you draw it if it's the first card you drew this turn.",
    "Modular": "This creature enters the battlefield with a number of +1/+1 counters. When it dies, you may put its +1/+1 counters on target artifact creature.",
    "Myriad": "Whenever this creature attacks, for each opponent other than defending player, you may create a token that's a copy of this creature that's tapped and attacking that opponent or a planeswalker they control. Exile the tokens at end of combat.",
    "Outlast": "Pay the outlast cost and tap this creature: Put a +1/+1 counter on it. Outlast only as a sorcery.",
    "Overload": "You may cast this spell for its overload cost. If you do, change its text by replacing all instances of 'target' with 'each.'",
    "Landfall": "Whenever a land enters the battlefield under your control, [effect].",
    "Partner with": "When this creature enters the battlefield, target player may put [partner name] into their hand from their library, then shuffle.",
    "Partner": "You can have two commanders if both have partner.",
    "Metalcraft": "This ability functions as long as you control three or more artifacts.",
    "Islandwalk": "This creature can't be blocked as long as defending player controls an Island.",
    "Treasure": "Artifact token with 'Tap, Sacrifice this artifact: Add one mana of any color.'",
    "Soulbond": "You may pair this creature with another unpaired creature when either enters the battlefield. They remain paired as long as you control both.",
    "Crew": "Tap any number of creatures you control with total power N or more: This Vehicle becomes an artifact creature until end of turn.",
    "Renown": "When this creature deals combat damage to a player, if it isn't renowned, put N +1/+1 counters on it and it becomes renowned.",
    "Escape": "You may cast this card from your graveyard for its escape cost. (Exile N other cards from your graveyard.)",
    "First strike": "This creature deals combat damage before creatures without first strike.",
    "Mill": "Put the top N cards of target player's library into their graveyard.",
    "Encore": "Exile this card from your graveyard: For each opponent, create a token copy that attacks that opponent this turn if able. They gain haste. Sacrifice them at the beginning of the next end step.",
    "Goad": "Until your next turn, that creature attacks each combat if able and attacks a player other than you if able.",
    "Hexproof from": "This permanent can't be the target of [specified quality] spells or abilities your opponents control.",
    "Umbra armor": "If enchanted creature would be destroyed, instead remove all damage from it and destroy this Aura.",
    "Fight": "Each creature deals damage equal to its power to the other.",
    "Reconfigure": "Attach to target creature you control; or unattach from a creature. Reconfigure only as a sorcery.",
    "Afflict": "Whenever this creature becomes blocked, defending player loses N life.",
    "Melee": "Whenever this creature attacks, it gets +1/+1 until end of turn for each opponent you attacked this combat.",
    "Living weapon": "When this Equipment enters the battlefield, create a 0/0 black Germ creature token, then attach this to it.",
    "Exhaust": "This ability is not a standard MTG keyword.",
    "Echo": "At the beginning of your upkeep, if this permanent came under your control since the beginning of your last upkeep, sacrifice it unless you pay its echo cost.",
    "Ward": "Whenever this permanent becomes the target of a spell or ability an opponent controls, counter it unless that player pays N.",
    "Multikicker": "You may pay an additional [cost] any number of times as you cast this spell.",
    "Split second": "As long as this spell is on the stack, players can't cast spells or activate abilities that aren't mana abilities.",
    "Improvise": "Your artifacts can help cast this spell. Each artifact you tap after you're done activating mana abilities pays for {1}.",
    "Monstrosity": "If this creature isn't monstrous, put N +1/+1 counters on it and it becomes monstrous.",
    "Plot": "You may pay [cost] and exile this card from your hand. Cast it as a sorcery on a later turn without paying its mana cost.",
    "Spree": "Choose one or more modes. As an additional cost to cast this spell, pay the costs associated with those modes.",
    "Changeling": "This card is every creature type.",
    "Gift": "You may cast this spell for its gift cost. If you do, [effect].",
    "Collect evidence": "Exile cards from your graveyard with total mana value N or greater.",
    "Surveil": "Look at the top N cards of your library. Put any number into your graveyard and the rest on top in any order.",
    "Investigate": "Create a Clue token. (It's an artifact with '2, Sacrifice this artifact: Draw a card.')",
    "Dethrone": "Whenever this creature attacks the player with the most life or tied for most life, put a +1/+1 counter on it.",
    "Food": "A Food is an artifact with '2, T: Sacrifice this artifact: You gain 3 life.'",
    "Disguise": "You may cast this card face down as a 2/2 creature with ward 2. Turn it face up any time for its disguise cost.",
    "Mentor": "Whenever this creature attacks, put a +1/+1 counter on target attacking creature with lesser power.",
    "Offspring": "When this creature dies, create a token that's a copy of it, except it's 1/1 and has no abilities.",
    "Mobilize": "Tap this creature: Untap another target creature.",
    "Suspect": "A suspected creature has menace and can't block.",
    "Double strike": "This creature deals both first-strike and regular combat damage.",
    "Proliferate": "Choose any number of permanents and/or players with counters. Give each another counter of a kind already there.",
    "Saddle": "Tap any number of creatures you control with total power N or more: This Mount becomes a creature until end of turn.",
    "Valiant": "Whenever this creature attacks alone, it gets +1/+1 until end of turn.",
    "Fabricate": "When this creature enters the battlefield, put N +1/+1 counters on it or create N 1/1 colorless Servo artifact creature tokens.",
    "Forage": "You may exile a card from your graveyard. If you do, [effect].",
    "Flanking": "Whenever a creature without flanking blocks this creature, the blocking creature gets -1/-1 until end of turn.",
    "Threshold": "As long as seven or more cards are in your graveyard, [effect].",
    "Flurry": "Whenever you cast a spell, [effect].",
    "Battle Cry": "Whenever this creature attacks, each other attacking creature gets +1/+0 until end of turn.",
    "Harmonize": "Draw three cards.",
    "Cloak": "Put the card onto the battlefield face down as a 2/2 creature with ward 2. Turn it face up any time for its mana cost if it's a creature card.",
    "Landcycling": "Pay [cost], discard this card: Search your library for a land card, reveal it, put it into your hand, then shuffle.",
    "Basic landcycling": "Pay [cost], discard this card: Search your library for a basic land card, reveal it, put it into your hand, then shuffle.",
    "Typecycling": "Pay [cost], discard this card: Search your library for a card of the specified type, reveal it, put it into your hand, then shuffle.",
    "Manifest": "Put the card onto the battlefield face down as a 2/2 creature. Turn it face up any time for its mana cost if it's a creature card.",
    "Lieutenant": "As long as you control your commander, [effect].",
    "Magecraft": "Whenever you cast or copy an instant or sorcery spell, [effect].",
    "Tempting offer": "When this spell resolves, each opponent may choose to copy it. For each opponent who does, [effect].",
    "Council's dilemma": "When this spell resolves, each player votes for one of two options. Then, [effect based on votes].",
    "Demonstrate": "When you cast this spell, you may copy it. If you do, choose an opponent to also copy it. The copies may choose new targets.",
    "Eternalize": "Pay [cost], exile this card from your graveyard: Create a token that's a copy of it, except it's a 4/4 black Zombie with no mana cost.",
    "Exert": "When this creature attacks, you may choose to exert it. If you do, it won't untap during your next untap step, and [effect].",
    "Riot": "This creature enters the battlefield with your choice of a +1/+1 counter or haste.",
    "Connive": "Draw a card, then discard a card. If you discarded a nonland card, put a +1/+1 counter on this creature.",
    "Channel": "Pay [cost], discard this card: [effect].",
    "Shadow": "This creature can block or be blocked only by creatures with shadow.",
}


def filter_commander_legal_cards(
    input_file, output_file, set_list, indent=1, output_dir=""
):
    """
    Process a Scryfall JSON file and create a new JSON with commander-legal cards from specified sets.

    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        set_list (list): List of set codes to include (e.g., ['drc', 'cmr'])
    """
    try:
        # Load the input JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            cards = json.load(f)

        # Filter cards
        filtered_cards = []
        missing_keywords = []
        for card in cards:
            # Skip if not in our set list or not commander legal
            if (
                card.get("set") not in set_list
                or card.get("legalities", {}).get("commander") != "legal"
            ):
                continue

            # Create new card object with only the fields we want
            filtered_card = {
                "name": card.get("name"),
                "set": card.get("set"),
                "mana_cost": card.get("mana_cost", ""),
                "cmc": card.get("cmc", 0),
                "type_line": card.get("type_line", ""),
                "oracle_text": card.get("oracle_text", ""),
                "power": card.get("power", ""),
                "toughness": card.get("toughness", ""),
                "colors": card.get("colors", []),
                "keywords": card.get("keywords", []),
            }

            # channge keyword field from list of string to list of object of the type {keyword: description}
            new_keywords = []
            for keyword in filtered_card["keywords"]:
                if keyword in mtg_keywords:
                    new_keywords.append({keyword: mtg_keywords[keyword]})
                else:
                    if keyword not in missing_keywords:
                        missing_keywords.append(keyword)

            filtered_card["keywords"] = new_keywords
            filtered_cards.append(filtered_card)

        match indent:
            case 0:
                save_cards(filtered_cards, output_file, output_dir, indent_bol=False)
            case 1:
                save_cards(filtered_cards, output_file, output_dir, indent_bol=True)
            case 2:
                save_cards(filtered_cards, output_file, output_dir, indent_bol=True)
                save_cards(filtered_cards, output_file, output_dir, indent_bol=False)

        # Save missing keywords to a separate file, if present already overwrite the file
        if missing_keywords:
            with open(output_dir + "missing_keywords.json", "w", encoding="utf-8") as f:
                json.dump(missing_keywords, f, indent=4, ensure_ascii=False)
        else:
            print("No missing keywords found.")

        print(f"Successfully processed {len(filtered_cards)} cards to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def save_cards(cards, output_file, output_dir, indent_bol=True):
    """
    Save the filtered cards to a new JSON file.

    Args:
        cards (list): List of filtered card dictionaries
        output_file (str): Path to output JSON file
        output_dir (str): Directory for output files
        indent (bool): Whether to use indentation in the JSON file
    """
    if indent_bol:
        with open(output_dir + "indent/" + output_file, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=4, ensure_ascii=False)
    else:
        with open(output_dir + "no_indent/" + output_file, "w", encoding="utf-8") as f:
            json.dump(cards, f, ensure_ascii=False)


if __name__ == "__main__":
    date = time.strftime("%Y%m%d%H%M%S")
    input_json = conf.bulk_file
    output_json = "commander_legal_cards" + date + ".json"  # Output file
    output_dir = "datasets/processed/"

    sets_to_include = conf.sets_to_include

    filter_commander_legal_cards(
        input_json, output_json, sets_to_include, indent=2, output_dir=output_dir
    )
