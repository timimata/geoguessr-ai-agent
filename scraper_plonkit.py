import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Todos os 90+ países oficialmente cobertos no GeoGuessr (e que o Plonkit costuma ter guia)
COUNTRIES_TO_SCRAPE = [
    # América do Norte e Central
    "canada", "united-states", "mexico", "guatemala", "costa-rica", "panama", 
    "dominican-republic", "puerto-rico", 
    # América do Sul
    "colombia", "ecuador", "peru", "bolivia", "brazil", "chile", "argentina", "uruguay",
    # Europa Ocidental e Sul
    "iceland", "faroe-islands", "ireland", "united-kingdom", "portugal", "spain", 
    "andorra", "france", "monaco", "italy", "san-marino", "malta",
    # Europa Central e Norte
    "belgium", "netherlands", "luxembourg", "germany", "switzerland", "austria", 
    "denmark", "norway", "sweden", "finland",
    # Europa Leste e Balcãs
    "poland", "czechia", "slovakia", "hungary", "slovenia", "croatia", "serbia", 
    "montenegro", "albania", "north-macedonia", "greece", "bulgaria", "romania", 
    "ukraine", "russian-federation", "estonia", "latvia", "lithuania",
    # Médio Oriente e Ásia Central
    "turkey", "israel", "jordan", "united-arab-emirates", "qatar", "kyrgyzstan",
    # Ásia do Sul e Leste
    "mongolia", "south-korea", "japan", "taiwan", "hong-kong", "macau", 
    "bangladesh", "bhutan", "india", "sri-lanka",
    # Sudeste Asiático
    "thailand", "cambodia", "laos", "malaysia", "singapore", "indonesia", "philippines",
    # África
    "senegal", "ghana", "nigeria", "tunisia", "kenya", "uganda", "rwanda", 
    "madagascar", "south-africa", "lesotho", "eswatini", "botswana",
    # Oceania
    "australia", "new-zealand", "greenland"
]

async def main():
    meta_dir = Path("metas")
    meta_dir.mkdir(exist_ok=True)
    
    print("A iniciar o Scraper automático do Plonkit...")
    
    async with async_playwright() as p:
        # Abrir o browser de forma invisível
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        sucesso = 0
        
        for country in COUNTRIES_TO_SCRAPE:
            url = f"https://www.plonkit.net/{country}"
            print(f"A extrair dicas para: {country.upper()}...")
            
            try:
                # Vai ao site do plonkit
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                # Espera 2 seg para a página renderizar o texto
                await asyncio.sleep(2)
                
                # Extrai todo o texto dos títulos, parágrafos e listas (Dicas Cruciais)
                # O Plonkit.net usa Notion por baixo, logo as classes notion-text e notion-list são comuns.
                # Como alternativa, apanhamos os P e LIs fundamentais.
                elements = await page.query_selector_all("p, li, h2, h3")
                
                extracted_text = []
                for el in elements:
                    text = await el.inner_text()
                    text = text.strip()
                    if len(text) > 15 and "Plonkit" not in text and text.lower() not in ["home", "guide", "about", "discord"]:
                        extracted_text.append("- " + text)
                
                if extracted_text:
                    # Guarda na pasta metas
                    safe_name = country.replace("-", " ") 
                    file_path = meta_dir / f"{safe_name}.txt"
                    
                    header = f"--- {safe_name.upper()} PLONKIT SCRAPED GEOHINTS ---\n"
                    # Junta as dicas (apenas as primeiras 30-40 linhas mais importantes para não rebentar a memória do LLM)
                    body = "\n".join(extracted_text[:40]) 
                    
                    file_path.write_text(header + body + "\n", encoding="utf-8")
                    print(f"  ✓ Informação guardada em {file_path.name}")
                    sucesso += 1
                else:
                    print(f"  ✗ Não foi possível encontrar texto para {country}")
                    
            except Exception as e:
                print(f"  ⚠ Erro a aceder a {country}: {e}")
                
        await browser.close()
        print(f"\nScraping Concluído! Extraídas enciclopédias detalhadas para {sucesso} países.")

if __name__ == "__main__":
    asyncio.run(main())