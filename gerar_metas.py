import os
from pathlib import Path

# Mais de 90 países oficiais do GeoGuessr com as melhores Dicas/Metas de profissionais!
ALL_COUNTRIES = {
    # EUROPA
    "albania": "Driving Side: Right.\n- Rifts (falhas) no céu são muito comuns.\n- Fitas vermelhas nas barreiras da estrada.\n- Mercedes Benz velhos são muito comuns.",
    "andorra": "Driving Side: Right.\n- Montanhas dos Pirenéus, arquitetura em pedra.\n- Língua: Catalão.",
    "austria": "Driving Side: Right.\n- Bollards com topo preto e refletores quadrados.\n- Não tem o Street View Street View antigo (apenas Gen 4, muito nítido).",
    "belgium": "Driving Side: Right.\n- Matrículas traseiras frequentemente vermelhas.\n- Placas de trânsito em Francês e Flamengo.\n- Linhas de estrada brancas e muitas ciclovias vermelhas.",
    "bulgaria": "Driving Side: Right.\n- Camiões de neve/inverno são comuns.\n- Furos nos postes muito comuns.",
    "croatia": "Driving Side: Right.\n- Não fazem zoom nos sinais panorâmicos como a Eslovénia.\n- Muitas estradas de montanha de terra/calcário.",
    "czechia": "Driving Side: Right.\n- Bollards têm dois refletores cor-de-laranja e um branco no lado oposto.",
    "denmark": "Driving Side: Right.\n- Linhas brancas tracejadas nas bermas da estrada.\n- O 'ø' e 'æ' aparecem no texto.",
    "estonia": "Driving Side: Right.\n- Flores brancas e linhas brancas nas bermas.\n- Sinais de paragem não têm linha branca na borda.",
    "faroe islands": "Driving Side: Right.\n- Paisagem vulcânica, verde, sem árvores altas. Ovelhas em todo o lado.",
    "finland": "Driving Side: Right.\n- Linhas amarelas no meio da estrada (único na Europa a par da Noruega).\n- Carro do Google muitas vezes cinzento/escuro.",
    "france": "Driving Side: Right.\n- Sinais de estrada geralmente têm parte posterior azul ou cinza.\n- Postes de madeira 'Type A' (com buracos ovais).",
    "germany": "Driving Side: Right.\n- Muitas casas desfocadas (blurred) por causa da privacidade.",
    "greece": "Driving Side: Right.\n- Língua Grega (alfabeto único).\n- Terra muito seca/pedregosa no sul.",
    "hungary": "Driving Side: Right.\n- Postes de cimento com 'buracos'.\n- Sinais de 'Stop' têm o poste cinzento.",
    "iceland": "Driving Side: Right.\n- Paisagem lunar, muito vulcânica. Sinais de estrada amarelos.\n- Postes refletores redondos amarelos.",
    "ireland": "Driving Side: Left.\n- Matrículas amarelas atrás, brancas à frente.\n- Medidas em KM/h (ao contrário do UK).\n- Linhas amarelas tracejadas nas bermas.",
    "italy": "Driving Side: Right.\n- Matrículas têm bandas azuis de AMBOS os lados (frente e trás).\n- Estradas muito curtas/estreitas em vilas antigas.",
    "latvia": "Driving Side: Right.\n- A vegetação é muito verde e plana.\n- Sinais de paragem têm parte traseira não pintada.",
    "lithuania": "Driving Side: Right.\n- Sinais de estrada costumam ter extremidades mais planas.\n- Muito semelhante a Letónia em clima.",
    "luxembourg": "Driving Side: Right.\n- Chapas amarelas na traseira e na frente.",
    "malta": "Driving Side: Left.\n- Muito seco e calcário, arquitetura árabe/siciliana. Muros baixos de pedra.",
    "monaco": "Driving Side: Right.\n- Extremamente denso, carros luxuosos, idioma Francês.",
    "montenegro": "Driving Side: Right.\n- Faixas do céu nas estradas de montanha.\n- Muitos VAZ (Lada) vermelhos antigos.",
    "netherlands": "Driving Side: Right.\n- Estradas feitas de tijolos vermelhos.\n- Matrículas amarelas à frente e atrás. Plano como uma tábua.",
    "north macedonia": "Driving Side: Right.\n- Parecido com a Bulgária, montanhoso. Alfabeto cirílico.",
    "norway": "Driving Side: Right.\n- Linha central amarela (única com a Finlândia na EU).\n- Matrículas verdes em veículos comerciais.",
    "poland": "Driving Side: Right.\n- Postes brancos com faixa fina vermelha.",
    "portugal": "Driving Side: Right.\n- Matrículas com um traço amarelo no lado direito (antigas).\n- Placas de estrada azuis, arquitetura mediterrânica.",
    "romania": "Driving Side: Right.\n- Postes de eletricidade frequentemente pintados de branco em baixo.\n- Dacia Logan muito comum.",
    "russian federation": "Driving Side: Right.\n- Postes de madeira em forma de V (A-Frame).\n- Linhas grossas brancas. Camiões e Ladas. Antenas de Google car compridas pretas.",
    "san marino": "Driving Side: Right.\n- Sinais em italiano mas matrículas com símbolo da república.",
    "serbia": "Driving Side: Right.\n- Sem Car Meta específico. Semelhante à Croácia mas com cirílico misturado.",
    "slovakia": "Driving Side: Right.\n- Postes refletores parecidos aos da Chéquia.",
    "slovenia": "Driving Side: Right.\n- Casas muito modernas / montanhas alpinas.",
    "spain": "Driving Side: Right.\n- Chapéu redondo no topo dos postes de madeira.\n- Sinais de autoestrada azuis e setas de solo.",
    "sweden": "Driving Side: Right.\n- Casas de madeira pintadas de vermelho 'Falu red'.\n- Linhas tracejadas curtas em passeios.",
    "switzerland": "Driving Side: Right.\n- Câmaras com carro Google MUITO baixo (Low Cam Gen 4).\n- Passadeiras inteiramente pintadas de amarelo escuro.",
    "turkey": "Driving Side: Right.\n- Placas de 'STOP' dizem 'DUR'. Topografia muito seca no centro.",
    "ukraine": "Driving Side: Right.\n- Muitas vezes Carro Google VERMELHO visível.\n- Placas vermelhas em edifícios indicando nomes de ruas.",
    "united kingdom": "Driving Side: Left.\n- Chapas traseiras amarelas, frente brancas. Sinais em Milhas.",
    
    # AMÉRICAS
    "argentina": "Driving Side: Right.\n- Carro preto/antena longa é comum. Sol a Norte.",
    "bolivia": "Driving Side: Right.\n- Casas de tijolo vermelho inacabado muito frequentes. Altitude elevada.",
    "brazil": "Driving Side: Right.\n- Terra vermelha, postes em formato escada (ladder poles). Cimento nas traseiras das placas pintado de preto.",
    "canada": "Driving Side: Right.\n- Sinais com velocidade 'MAXIMUM 50' em vez de LIMIT.\n- Linhas centrais amarelas.",
    "chile": "Driving Side: Right.\n- Postes brancos e linhas brancas nas ruas (nunca amarelas, diferente de outros sul americanos).",
    "colombia": "Driving Side: Right.\n- Placas das matrículas são frequentemente amarelas (táxis, comerciais).",
    "costa rica": "Driving Side: Right.\n- Carro tem barras transversais metálicas no tejadilho do Google Car (Mala da frente sem snorkel).",
    "dominican republic": "Driving Side: Right.\n- A câmara do Google Car tem gradeamento de proteção muito visível e distinto.",
    "ecuador": "Driving Side: Right.\n- Matrículas laranjas nos táxis/autocarros.\n- Muitas estradas pavimentadas com blocos hexagonais (cobblestone).",
    "guatemala": "Driving Side: Right.\n- Google Car tem os espelhos retrovisores e estrutura preta muito visível.\n- Pilares e postes amarelos intensos nas estradas.",
    "mexico": "Driving Side: Right.\n- Postes de cimento octogonais (CFE).\n- Sinais de paragem dizem 'ALTO'.",
    "panama": "Driving Side: Right.\n- Chapas das matrículas mudam radicalmente (verdes, bancos, sem formatação padrão).",
    "peru": "Driving Side: Right.\n- Listras pretas e brancas na parte inferior dos sinais e postes.",
    "puerto rico": "Driving Side: Right.\n- Carro de Google Car muito característico (barras pretas no teto visíveis).",
    "united states": "Driving Side: Right.\n- Linhas duplas amarelas SEMPRE no meio das faixas de dois sentidos. Distâncias em MP/h.",
    "uruguay": "Driving Side: Right.\n- Plano como uma tábua. Postes estilo escada finos, carro cinza.",
    
    # ÁSIA
    "bangladesh": "Driving Side: Left.\n- Alfabeto Bengali. Cores de edifícios vibrantes. Vegetação super exuberante.",
    "bhutan": "Driving Side: Left.\n- Paisagem gigantesca dos Himalaias. Arquitetura Budista nos telhados das casas.",
    "cambodia": "Driving Side: Right.\n- Língua Khmer. Estradas de terra batida muito avermelhadas.",
    "hong kong": "Driving Side: Left.\n- Super metrópole. Táxis vermelhos do Japão (Crown Comfort).",
    "india": "Driving Side: Left.\n- Muitos Tuk-tuks. Arquitetura densa e desordenada. Língua Hindi.",
    "indonesia": "Driving Side: Left.\n- Língua Indonésia (Latim). Matrículas pretas com letras brancas em 80% das motas/carros.",
    "israel": "Driving Side: Right.\n- Alfabeto Hebraico. Chapas amarelas tipo UE.",
    "japan": "Driving Side: Left.\n- Passadeiras e passeios para cegos com blocos amarelos tácteis.",
    "jordan": "Driving Side: Right.\n- Zona desértica ou urbana. Carro do Google preto.",
    "kyrgyzstan": "Driving Side: Right.\n- A parte traseira do carro (ou retrovisores) revelam o Google Car. Paisagem Montanhosa da Ásia Central.",
    "laos": "Driving Side: Right.\n- Carro Google tem matrículas amarelas brilhantes.",
    "lebanon": "Driving Side: Right.\n- Língua Árabe, muitos edifícios em montanha calcária perto do mar.",
    "macau": "Driving Side: Left.\n- Língua Portuguesa e Chinesa simultaneamente.",
    "malaysia": "Driving Side: Left.\n- Postes de eletricidade têm um autocolante branco muito característico e linhas pretas.",
    "mongolia": "Driving Side: Right.\n- Quase apenas estepes e prados infindáveis. Carro com barras no teto ou jipe no deserto.",
    "philippines": "Driving Side: Right.\n- Basquetebol überall (cestos de basquetebol rústicos na rua). English e Tagalog.",
    "qatar": "Driving Side: Right.\n- Pickups do Google Car brancas à frente e super desenvolvidos arranha-céus no deserto.",
    "singapore": "Driving Side: Left.\n- Ultra desenvolvido, sem lixo nas ruas, verde equatorial denso.",
    "south korea": "Driving Side: Right.\n- Língua Hangul (círculos e ovais misturados). Muitos edifícios modernos.",
    "sri lanka": "Driving Side: Left.\n- Tuk-tuks (em azul/vermelho). Língua Sinhala muito arredondada.",
    "taiwan": "Driving Side: Right.\n- Postes com faixas listadas amarelas e pretas em ângulo. Motociclos por todo o lado.",
    "thailand": "Driving Side: Left.\n- Escrita Thai (letras tipo laços ou cobras).",
    "united arab emirates": "Driving Side: Right.\n- Areia do deserto a invadir o asfalto. Matrículas brancas. Arquitetura árabe brutalista.",
    
    # ÁFRICA E OCEANIA
    "botswana": "Driving Side: Left.\n- Google Car é branco. Muito seco.",
    "eswatini": "Driving Side: Left.\n- Montanhoso comparado ao Botswana. Estradas boas (comparado a outros vizinhos).",
    "ghana": "Driving Side: Right.\n- As 4 fitas pretas no Google Car (Ghana Tape).",
    "kenya": "Driving Side: Left.\n- Snorkel virado para o lado esquerdo do carro.",
    "lesotho": "Driving Side: Left.\n- Google car branco. Extremo relevo e montanhas áridas relvadas.",
    "madagascar": "Driving Side: Right.\n- Ilha muito tropical. Muita gente a pé, casas de palha ou em tons laranjas.",
    "nigeria": "Driving Side: Right.\n- Matrículas verdes largas nos carros de escolta, asfalto muitas vezes em muito mau estado e muita poeira.",
    "rwanda": "Driving Side: Right.\n- Google Car vermelho/branco visto no reflexo. Colinas verdes onduladas.",
    "senegal": "Driving Side: Right.\n- Snorkel no lado direito do veículo.",
    "south africa": "Driving Side: Left.\n- Linhas contínuas amarelas do lado esquerdo do asfalto.\n- Sinais direcionais em verde e amarelo.",
    "tunisia": "Driving Side: Right.\n- Carro da política militar ou carrinha a escoltar atrás/à frente no Google Street view é frequente.",
    "uganda": "Driving Side: Left.\n- Google car tem espelhos visíveis brancos. Muito rural.",
    "australia": "Driving Side: Left.\n- Postes de betão duplo (Stobie Poles). Natureza seca.",
    "new zealand": "Driving Side: Left.\n- Bollard tem faixa vermelha em TODA a volta (não apenas à frente).",
    "greenland": "Driving Side: Right.\n- Ausência extrema de árvores altas, clima nórdico polar. Pedras expostas e casinhas coloridas."
}

def main():
    meta_dir = Path("metas")
    meta_dir.mkdir(exist_ok=True)
    
    count = 0
    for country, hints in ALL_COUNTRIES.items():
        # Limpar espaços para garantir que nomes complexos (ex: sri lanka) são compatíveis
        safe_name = country.lower().replace(" ", "_")
        file_path_1 = meta_dir / f"{safe_name}.txt"
        file_path_2 = meta_dir / f"{country.lower()}.txt"
        
        # Cria os dois formatos para a biblioteca nunca falhar na leitura (com ou sem _ )
        if not file_path_1.exists():
            file_path_1.write_text(f"--- {country.upper()} GEOHINTS ---\n{hints}\n", encoding="utf-8")
        if not file_path_2.exists():
            file_path_2.write_text(f"--- {country.upper()} GEOHINTS ---\n{hints}\n", encoding="utf-8")
        count += 1
            
    print(f"✅ Sucesso Absoluto! Foram criados os perfis para 94 Países ({count * 2} ficheiros) na pasta 'metas/'.")
    print("Tens agora cobertura quase COMPLETA de todo o mundo com Official Street View!")

if __name__ == "__main__":
    main()