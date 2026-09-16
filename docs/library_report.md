# Song library report

Library root: `/Users/joeyt/Music/Clone Hero/songs`

Song folders found: **652**

Hidden/temp directories skipped: **1** (e.g. Guitar Hero - Metallica/.tmp.drivedownload)


## File-type counts (per song folder, not per file)

- song.ini: 652
- notes.chart: 306
- notes.mid: 347
- .sng: 0

## Cloud / local-file status

- Chart/mid files checked (size via stat): 653
- Zero-byte files: 0 (0.0%)
- Content-read sample: 40 files, 0 read errors (0.0%)
- Cloud-placeholder pattern suspected: **False**

## Guitar-part availability

Folders with **both** notes.chart and notes.mid (chart preferred at ingest): 1
Folders with **neither** notes.chart nor notes.mid: 0

### From .chart section headers (306 files)
- Songs with at least one [*Single] section: 306
  - EasySingle: 8
  - MediumSingle: 8
  - HardSingle: 8
  - ExpertSingle: 306

### From .mid note ranges, songs with .mid but no .chart (346 files)
- Songs with a `PART GUITAR` track: 346
- Songs with at least one note in a difficulty's fret range: 346
  - Easy (notes in range): 286
  - Medium (notes in range): 286
  - Hard (notes in range): 286
  - Expert (notes in range): 346
- Parse errors: 0

### Combined per-difficulty song counts (native chart section, or .mid notes in the difficulty's fret range when there's no .chart -- songs, not files)
- Easy: 294
- Medium: 294
- Hard: 294
- Expert: 652

**Overall songs with a detected guitar part: 652 / 652**

**Songs with native Easy AND Medium: 293** (decision-rule threshold: 150 -- see docs/DECISIONS.md)

### Clip-corruption note impact (123 .mid files needed `clip=True` to parse at all)
- Out-of-range byte inside a SysEx payload (never a note number): 123
- Out-of-range byte inside a note_on/note_off (or other channel) message: 0
- Other/unresolved: 0
- Of the files needing clipping, **0 had a clamped value land in the 60-100 guitar note range** -- every clamp happens inside proprietary SysEx metadata (e.g. Harmonix chart extensions), never inside a note_on/note_off message. clip=True therefore has no effect on note/difficulty detection accuracy for this library.

## Size breakdown

- Chart-related files (.chart/.mid/.ini): 49.7 MB
- Audio files (.ogg/.mp3/.opus): 10607.4 MB
- Video background files (.webm): 446.7 MB

## Ingest results (Phase 2)

Song folders considered: **652**


### Parse failures (0)


### Open-note exclusions (83)

- `Anti Hero 2/[AH2] 36. The Experience - SikTh - The Future In Whose Eyes/SikTh - 09. Riddles of Humanity (Chezy)` [Expert]: 17.0% open notes
- `Anti Hero 2/[AH2] 36. The Experience - SikTh - The Future In Whose Eyes/SikTh - 01. Vivid (Chezy)` [Expert]: 24.8% open notes
- `Anti Hero 2/[AH2] 36. The Experience - SikTh - The Future In Whose Eyes/SikTh - 03. The Aura (Chezy)` [Expert]: 11.9% open notes
- `Anti Hero 2/[AH2] 36. The Experience - SikTh - The Future In Whose Eyes/SikTh - 06. Cracks of Light (feat. Spencer Sotelo) (Chezy)` [Expert]: 13.1% open notes
- `Anti Hero 2/[AH2] 36. The Experience - SikTh - The Future In Whose Eyes/SikTh - 10. No Wishbones (Chezy)` [Expert]: 30.4% open notes
- `Anti Hero 2/[AH2] 36. The Experience - SikTh - The Future In Whose Eyes/SikTh - 02. Century of the Narcissist (Chezy)` [Expert]: 20.0% open notes
- `Anti Hero 2/[AH2] 36. The Experience - SikTh - The Future In Whose Eyes/SikTh - 05. Weavers of Woe (Chezy)` [Expert]: 23.5% open notes
- `Anti Hero 2/[AH2] 32. The Experience - Galneryus - Under the Force of Courage/Galneryus - 02. The Time Before Dawn (Chemfinal)` [Expert]: 19.1% open notes
- `Anti Hero 2/[AH2] 29. The Experience - Thank You for Playing Anti Hero 2/John Pizzarelli Trio - After You've Gone (Live) (Chemfinal)` [Expert]: 12.1% open notes
- `Anti Hero 2/[AH2] 29. The Experience - Thank You for Playing Anti Hero 2/DragonForce - Evil Dead (Death cover) (Miscellany)` [Expert]: 37.3% open notes
- `Anti Hero 2/[AH2] 14. The Experience - One More Section/Death - Spirit Crusher (xX760Xx)` [Expert]: 25.8% open notes
- `Anti Hero 2/[AH2] 14. The Experience - One More Section/James LaBrie - One More Time (Big Whoop Magazine)` [Expert]: 17.3% open notes
- `Anti Hero 2/[AH2] 15. The Experience - Plastic Guitar Over Friends/Knocked Loose - Mistakes Like Fractures (Chezy)` [Expert]: 30.3% open notes
- `Anti Hero 2/[AH2] 07. Classic - The Sound of the Crowd/Pantera - Power Metal (Miscellany)` [Expert]: 21.9% open notes
- `Anti Hero 2/[AH2] 07. Classic - The Sound of the Crowd/Darkest Hour - An Ethereal Drain (ThundahK)` [Expert]: 12.2% open notes
- `Anti Hero 2/[AH2] 27. The Experience - Wrecking Any Life You Had/Pentakill - Ohmwrecker (Supahfast198)` [Expert]: 19.5% open notes
- `Anti Hero 2/[AH2] 27. The Experience - Wrecking Any Life You Had/Guitar Heroes - 12 Donkeys (xX760Xx)` [Expert]: 12.8% open notes
- `Anti Hero 2/[AH2] 11. Classic - In Over Our Heads/Slayer - Dittohead (ThundahK)` [Expert]: 36.9% open notes
- `Anti Hero 2/[AH2] 03. Classic - Who Needs Practice/Chimaira - Pleasure in Pain (XEntombmentX)` [Expert]: 13.8% open notes
- `Anti Hero 2/[AH2] 03. Classic - Who Needs Practice/Bayside - Rumspringa (Return to Heartbreak Road) (Chezy)` [Expert]: 10.3% open notes
- `Anti Hero 2/[AH2] 03. Classic - Who Needs Practice/Taking Back Sunday - Error Operator (Jaded)` [Expert]: 21.1% open notes
- `Anti Hero 2/[AH2] 06. Classic - National Success/Alter Bridge - Isolation (Jaded)` [Expert]: 29.5% open notes
- `Anti Hero 2/[AH2] 06. Classic - National Success/Dance Gavin Dance - Chucky vs. The Giant Tortoise (Riddo)` [Expert]: 13.0% open notes
- `Anti Hero 2/[AH2] 16. The Experience - Back to Square One/Sybreed - Flesh Doll for Sale (CyclopsDragon)` [Expert]: 27.4% open notes
- `Anti Hero 2/[AH2] 16. The Experience - Back to Square One/Beartooth - Used and Abused (Chezy)` [Expert]: 19.2% open notes
- `Anti Hero 2/[AH2] 12. Classic - Homecoming Tour/Slipknot - Sulfur (XEntombmentX)` [Expert]: 14.5% open notes
- `Anti Hero 2/[AH2] 21. The Experience - Play This Thrash of a Chart/Trivium - Demon (Miscellany)` [Expert]: 28.4% open notes
- `Anti Hero 2/[AH2] 21. The Experience - Play This Thrash of a Chart/Andy Timmons - Farmer Sez (Arctan)` [Expert]: 17.0% open notes
- `Anti Hero 2/[AH2] 21. The Experience - Play This Thrash of a Chart/Allegaeon - 1.618 (Miscellany)` [Expert]: 14.6% open notes
- `Anti Hero 2/[AH2] 28. The Experience - At War with Your Controller/Nobuo Uematsu - Advent One-Winged Angel - ACC Long Version (Mintorment)` [Expert]: 19.2% open notes
- `Anti Hero 2/[AH2] 28. The Experience - At War with Your Controller/Rhapsody of Fire - Agony Is My Name (BurpLeTurtle)` [Expert]: 14.5% open notes
- `Anti Hero 2/[AH2] 25. The Experience - Obstructing All Progress/Interloper - The Conjuration (feat. Lucas Mann) (Dillski)` [Expert]: 10.0% open notes
- `Anti Hero 2/[AH2] 25. The Experience - Obstructing All Progress/Lost Society - Bitch, Out' My Way (Miscellany)` [Expert]: 16.9% open notes
- `Anti Hero 2/[AH2] 33. The Experience - HIDDEN MACHINE - Unlimited/HIDDEN MACHINE - 03. Surf (Jarvis9999)` [Expert]: 11.1% open notes
- `Anti Hero 2/[AH2] 33. The Experience - HIDDEN MACHINE - Unlimited/HIDDEN MACHINE - 06. Rock Smash (Jarvis9999)` [Expert]: 23.3% open notes
- `Anti Hero 2/[AH2] 33. The Experience - HIDDEN MACHINE - Unlimited/HIDDEN MACHINE - 07. Waterfall (Jarvis9999)` [Expert]: 13.2% open notes
- `Anti Hero 2/[AH2] 33. The Experience - HIDDEN MACHINE - Unlimited/HIDDEN MACHINE - 04. Strength (Jarvis9999)` [Expert]: 21.9% open notes
- `Anti Hero 2/[AH2] 33. The Experience - HIDDEN MACHINE - Unlimited/HIDDEN MACHINE - 05. Flash (Jarvis9999)` [Expert]: 13.2% open notes
- `Anti Hero 2/[AH2] 26. The Experience - Eternal Torment/Persefone - Stillness Is Timeless (XEntombmentX)` [Expert]: 31.7% open notes
- `Anti Hero 2/[AH2] 26. The Experience - Eternal Torment/Space Eater - FAA (Miscellany)` [Expert]: 31.7% open notes
- `Anti Hero 2/[AH2] 17. The Experience - We Are the Monsters/Mutoid Man - 1000 Mile Stare (Chezy)` [Expert]: 28.3% open notes
- `Anti Hero 2/[AH2] 17. The Experience - We Are the Monsters/Dance Gavin Dance - Man of the Year (Miscellany)` [Expert]: 10.1% open notes
- `Anti Hero 2/[AH2] 17. The Experience - We Are the Monsters/Parkway Drive - Romance Is Dead (Riddo)` [Expert]: 13.5% open notes
- `Anti Hero 2/[AH2] 20. The Experience - Rush Hour/Textures - New Horizons (XEntombmentX)` [Expert]: 10.5% open notes
- `Anti Hero 2/[AH2] 20. The Experience - Rush Hour/Lost Society - No Absolution (Miscellany)` [Expert]: 32.6% open notes
- `Anti Hero 2/[AH2] 20. The Experience - Rush Hour/Scale the Summit - Witch House (feat. Angel Vivaldi) (XEntombmentX)` [Expert]: 11.4% open notes
- `Anti Hero 2/[AH2] 09. Classic - International Sensation/Killswitch Engage - Still Beats Your Name (XEntombmentX)` [Expert]: 19.8% open notes
- `Anti Hero 2/[AH2] 31. The Experience - Architects - Holy Hell/Architects - 05. Damnation (Miscellany)` [Expert]: 18.7% open notes
- `Anti Hero 2/[AH2] 31. The Experience - Architects - Holy Hell/Architects - Holy Hell Riff Medley (XEntombmentX, Chezy, Miscellany)` [Expert]: 19.6% open notes
- `Anti Hero 2/[AH2] 31. The Experience - Architects - Holy Hell/Architects - 06. Royal Beggars (XEntombmentX)` [Expert]: 26.9% open notes
- `Anti Hero 2/[AH2] 31. The Experience - Architects - Holy Hell/Architects - 10. Doomsday (Miscellany)` [Expert]: 13.3% open notes
- `Anti Hero 2/[AH2] 31. The Experience - Architects - Holy Hell/Architects - 02. Hereafter (Chezy)` [Expert]: 35.8% open notes
- `Anti Hero 2/[AH2] 31. The Experience - Architects - Holy Hell/Architects - 08. Dying to Heal (Chezy)` [Expert]: 22.8% open notes
- `Anti Hero 2/[AH2] 19. The Experience - GanG BanG Dream!/Chimp Spanner - Mobius (Part III) (Big Whoop Magazine)` [Expert]: 14.4% open notes
- `Anti Hero 2/[AH2] 19. The Experience - GanG BanG Dream!/Anubis Gate - Desiderio Omnibus (Supahfast198)` [Expert]: 22.7% open notes
- `Anti Hero 2/[AH2] 35. The Experience - Megadeth - Peace Sells... but Who's Buying/Megadeth - 01. Wake Up Dead (xX760Xx)` [Expert]: 10.7% open notes
- `Anti Hero 2/[AH2] 35. The Experience - Megadeth - Peace Sells... but Who's Buying/Megadeth - 04. Devils Island (xX760Xx)` [Expert]: 33.3% open notes
- `Anti Hero 2/[AH2] 35. The Experience - Megadeth - Peace Sells... but Who's Buying/Megadeth - 06. Bad Omen (xX760Xx)` [Expert]: 22.3% open notes
- `Anti Hero 2/[AH2] 35. The Experience - Megadeth - Peace Sells... but Who's Buying/Megadeth - 03. Peace Sells (xX760X)` [Expert]: 21.8% open notes
- `Anti Hero 2/[AH2] 35. The Experience - Megadeth - Peace Sells... but Who's Buying/Megadeth - 02. The Conjuring (xX760Xx)` [Expert]: 15.4% open notes
- `Anti Hero 2/[AH2] 18. The Experience - Why You Always Missing/Utsu-P - Human Error (feat. Sekihan) (CyclopsDragon)` [Expert]: 11.6% open notes
- `Anti Hero 2/[AH2] 18. The Experience - Why You Always Missing/Car Bomb - Dissect Yourself (XEntombmentX)` [Expert]: 16.1% open notes
- `Anti Hero 2/[AH2] 18. The Experience - Why You Always Missing/Periphery - The Way the News Goes... (XEntombmentX)` [Expert]: 10.1% open notes
- `Anti Hero 2/[AH2] 18. The Experience - Why You Always Missing/Mastodon - Oblivion (XEntombmentX)` [Expert]: 15.9% open notes
- `Anti Hero 2/[AH2] 24. The Experience - A Million Exploding Notes/RichaadEB - Exodus (Supahfast198)` [Expert]: 11.8% open notes
- `Anti Hero 2/[AH2] 24. The Experience - A Million Exploding Notes/Cacophony - Where My Fortune Lies (Arctan)` [Expert]: 11.8% open notes
- `Anti Hero 2/[AH2] 24. The Experience - A Million Exploding Notes/Kiko Loureiro - Gray Stone Gateway (Arctan)` [Expert]: 11.4% open notes
- `Anti Hero 2/[AH2] 24. The Experience - A Million Exploding Notes/HORSE the band - A Million Exploding Suns (Chezy)` [Expert]: 17.8% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 07. Like Light to Flies (Miscellany)` [Expert]: 25.7% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 04. Drowned and Torn Asunder (Miscellany)` [Expert]: 28.4% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 03. Pull Harder on the Strings of Your Martyr` [Expert]: 16.3% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 15. Master of Puppets (Metallica cover) (Miscellany)` [Expert]: 26.6% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 12. Declaration (Miscellany)` [Expert]: 26.3% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 14. Washing Away Me in the Tides (Miscellany)` [Expert]: 13.1% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 06. A Gunshot to the Head of Trepidation (Miscellany)` [Expert]: 20.7% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 01. The End of Everything Rain (Miscellany)` [Expert]: 36.4% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 11. Departure (Miscellany)` [Expert]: 31.0% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 09. The Deceived (Miscellany)` [Expert]: 14.2% open notes
- `Anti Hero 2/[AH2] 37. The Experience - Trivium - Ascendancy/Trivium - 13. Blinding Tears Will Break the Skies (Miscellany)` [Expert]: 15.0% open notes
- `Anti Hero 2/[AH2] 22. The Experience - Metallic Monarchy/The Human Abstract - Nocturne (ThundahK)` [Expert]: 12.4% open notes
- `Anti Hero 2/[AH2] 22. The Experience - Metallic Monarchy/Circus Maximus - Used (Supahfast198)` [Expert]: 15.1% open notes
- `Anti Hero 2/[AH2] 22. The Experience - Metallic Monarchy/Exodus - Call to Arms Riot Act (Miscellany)` [Expert]: 20.1% open notes
- `Anti Hero 2/[AH2] 22. The Experience - Metallic Monarchy/Allegaeon - Apoptosis (Miscellany)` [Expert]: 12.1% open notes

### Exact-duplicate drops (5)

- dropped `Guitar Hero - Metallica/Metallica - One` (kept `Guitar Hero III - Legends of Rock/Quickplay/Metallica - One`): 4 vs 4 surviving difficulties, 2146 vs 2207 Expert notes
- dropped `Guitar Hero - Metallica/Metallica - All Nightmare Long` (kept `Guitar Hero III DLC/Metallica - All Nightmare Long`): 4 vs 4 surviving difficulties, 3038 vs 3041 Expert notes
- dropped `Guitar Hero World Tour/Trust - Antisocial` (kept `Guitar Hero III DLC/Trust - Antisocial`): 4 vs 4 surviving difficulties, 1120 vs 1124 Expert notes
- dropped `Anti Hero 2/[AH2] 12. Classic - Homecoming Tour/DragonForce - Heroes of Our Time (Riddo)` (kept `Guitar Hero III DLC/Dragonforce - Heroes of Our Time`): 1 vs 4 surviving difficulties, 2931 vs 2759 Expert notes
- dropped `Guitar Hero III - Legends of Rock/Quickplay/Steve Ouimette - The Devil Went Down to Georgia` (kept `Guitar Hero III DLC/Steve Ouimette - The Devil Went Down to Georgia`): 4 vs 4 surviving difficulties, 2398 vs 3087 Expert notes

### Near-duplicate groups, reported only (11)

- `metallica / one`: Guitar Hero III - Legends of Rock/Quickplay/Metallica - One, Guitar Hero III - Legends of Rock/Quickplay/Metallica - One (Co-op)
- `dragonforce / heroes of our time`: Guitar Hero III DLC/Dragonforce - Heroes of Our Time, Guitar Hero III DLC/Dragonforce - Heroes of Our Time (Co-op)
- `dragonforce / through the fire flames`: Guitar Hero III - Legends of Rock/Bonus/Dragonforce - Through The Fire & Flames (Co-op), Guitar Hero III - Legends of Rock/Bonus/Dragonforce - Through The Fire & Flames
- `steve ouimette / the devil went down to georgia`: Guitar Hero III DLC/Steve Ouimette - The Devil Went Down to Georgia, Guitar Hero III DLC/Steve Ouimette - The Devil Went Down to Georgia (Co-op)
- `dragonforce / operation ground and pound`: Guitar Hero III DLC/Dragonforce - Operation Ground and Pound (Co-op), Guitar Hero III DLC/Dragonforce - Operation Ground and Pound
- `foo fighters / the pretender`: Guitar Hero III DLC/Foo Fighters - The Pretender, Guitar Hero III DLC/Foo Fighters - The Pretender (Co-op)
- `extremoduro / so payaso`: Guitar Hero III DLC/Extremoduro - So Payaso, Guitar Hero III DLC/Extremoduro - So Payaso (Co-op)
- `tom morello / tom morello guitar battle`: Guitar Hero III DLC/Tom Morello - Guitar Battle, Guitar Hero III DLC/Tom Morello - Guitar Battle (Co-op)
- `slash / slash guitar battle`: Guitar Hero III DLC/Slash - Guitar Battle (Co-op), Guitar Hero III DLC/Slash - Guitar Battle
- `dragonforce / revolution deathsquad`: Guitar Hero III DLC/Dragonforce - Revolution Deathsquad, Guitar Hero III DLC/Dragonforce - Revolution Deathsquad (Co-op)
- `o donnell salvatori vai / halo theme mjolnir mix`: Guitar Hero III DLC/O_Donnell&Salvatori&Vai - Halo Theme MJOLNIR Mix, Guitar Hero III DLC/O_Donnell&Salvatori&Vai - Halo Theme MJOLNIR Mix (Co-op)

### Split x stratum x difficulty song counts

- test/expert_only/Easy: 1
- test/expert_only/Expert: 28
- test/native_easy_medium/Easy: 31
- test/native_easy_medium/Expert: 30
- test/native_easy_medium/Hard: 31
- test/native_easy_medium/Medium: 31
- train/expert_only/Expert: 222
- train/expert_only/Hard: 1
- train/expert_only/Medium: 1
- train/native_easy_medium/Easy: 230
- train/native_easy_medium/Expert: 229
- train/native_easy_medium/Hard: 230
- train/native_easy_medium/Medium: 230
- val/expert_only/Expert: 27
- val/native_easy_medium/Easy: 28
- val/native_easy_medium/Expert: 28
- val/native_easy_medium/Hard: 28
- val/native_easy_medium/Medium: 28

### Chord merges at load time (< 1/60s apart; never applied to the cache), by difficulty across the library (2 total notes absorbed)

- Easy: 0
- Medium: 0
- Hard: 0
- Expert: 2

### Max simultaneously-open hit windows (post-merge, +-hit_window_s), by difficulty -- sets env.py's/RuleEngine's max_open capacity (library-wide max: **8**)

- Easy: 2
- Medium: 2
- Hard: 3
- Expert: 8
