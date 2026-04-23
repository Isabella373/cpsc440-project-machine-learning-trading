SP500_CANDIDATES = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","BRK-B","LLY",
    "JPM","AVGO","WMT","V","MA","XOM","UNH","JNJ","PG","HD",
    "COST","MRK","ABBV","PEP","KO","ADBE","CSCO","CRM","TMO","MCD",
    "ACN","LIN","ORCL","NFLX","AMD","QCOM","INTC","TXN","AMAT","MU",
    "NOW","IBM","GE","CAT","HON","RTX","BA","DE","UPS","LMT",
    "SPGI","BLK","GS","MS","BAC","WFC","C","AXP","SCHW","BK",
    "ISRG","VRTX","REGN","GILD","PFE","BMY","ZTS","MDT","SYK","CI",
    "PLTR","SNPS","CDNS","KLAC","ADI","LRCX","NXPI","MCHP","ON","MPWR",
    "PANW","CRWD","ZS","FTNT","DDOG","OKTA","NET","TEAM","MDB","SNOW",
    "UBER","ABNB","DASH","BKNG","EXPE","EBAY","ETSY","RCL","CCL","NCLH",
    "LOW","TJX","TGT","DG","DLTR","ROST","ULTA","SBUX","YUM","CMG",
    "DIS","PARA","WBD","FOX","FOXA","NWSA","NWS","CHTR","TMUS","VZ",
    "T","D","AEP","SO","DUK","EXC","XEL","ED","PEG","EIX",
    "NEE","PCG","SRE","DTE","WEC","PPL","ES","FE","AES","NRG",
    "PSA","PLD","AMT","CCI","EQIX","DLR","SBAC","SPG","O","VICI",
    "EQR","AVB","UDR","ESS","MAA","INVH","KIM","REG","FRT","BXP",
    "KMB","CL","GIS","HSY","KHC","SJM","CAG","CPB","HRL","MKC",
    "KR","WBA","CVS","EL","STZ","MO","PM","BTI","TAP","BF-B",
    "HAL","SLB","BKR","COP","EOG","MPC","PSX","VLO","OXY","DVN",
    "APA","FANG","HES","KMI","WMB","OKE","TRGP","ENB","EPD","MRO",
    "NEM","FCX","NUE","STLD","X","AA","CMC","RS","ATI","PKG",
    "IP","WRK","BALL","SEE","SON","CF","MOS","NTR","LYB","DOW",
    "PPG","APD","ECL","SHW","ALB","FMC","IFF","CE","DD","EMN",
    "PYPL","SQ","COIN","HOOD","IBKR","AXON","ARES","KKR","APO","BX",
    "TTD","APP","SNDK","WDAY","ANET","SMCI","HPE","HPQ","DELL","NTAP",
    "STX","WDC","LULU","NKE","RL","TPR","VFC","PVH","CPRI","UAA"
]

# Map each ticker to a broad sector bucket for the sector feature
SUBSECTOR_MAP = {

    # Semiconductor
    **{t: "Semiconductor" for t in [
        "NVDA","AMD","AVGO","TXN","QCOM","INTC","AMAT","MU",
        "KLAC","ADI","LRCX","NXPI","MCHP","ON","MPWR"
    ]},

    # Software / IT Services
    **{t: "Software" for t in [
        "MSFT","CRM","ADBE","ORCL","NOW","SNPS","CDNS",
        "PANW","CRWD","ZS","FTNT","DDOG","OKTA","TEAM",
        "MDB","SNOW","WDAY","NET","TTD","PLTR","APP","ACN"
    ]},

    # Internet / Platforms
    **{t: "InternetPlatform" for t in [
        "GOOGL","GOOG","META","NFLX","UBER","DASH","ABNB"
    ]},

    # Hardware / Devices / Storage / Networking
    **{t: "Hardware" for t in [
        "AAPL","HPQ","DELL","HPE","IBM","CSCO","NTAP",
        "STX","WDC","ANET","SMCI","SNDK"
    ]},

    # Fintech
    **{t: "Fintech" for t in [
        "PYPL","SQ","COIN","HOOD","IBKR"
    ]},

    # E-commerce / Marketplace
    **{t: "Ecommerce" for t in [
        "AMZN","EBAY","ETSY"
    ]},

    # Travel / Consumer Platforms
    **{t: "ConsumerPlatform" for t in [
        "BKNG","EXPE","RCL","CCL","NCLH"
    ]},

    # Auto / EV
    **{t: "AutoEV" for t in [
        "TSLA"
    ]},

    # Retail / Restaurants / Apparel
    **{t: "RetailRestaurant" for t in [
        "HD","MCD","NKE","LOW","TJX","TGT","DG","DLTR",
        "ROST","ULTA","SBUX","YUM","CMG","LULU",
        "RL","TPR","VFC","PVH","CPRI","UAA"
    ]},

    # Financials
    **{t: "FinancialServices" for t in [
        "JPM","V","MA","BAC","GS","MS","BLK","AXP","SCHW",
        "C","WFC","BK","SPGI","ARES","KKR","APO","BX","BRK-B"
    ]},

    # Healthcare
    **{t: "Healthcare" for t in [
        "UNH","JNJ","MRK","ABBV","TMO","ISRG","VRTX","REGN",
        "GILD","PFE","BMY","ZTS","MDT","SYK","CI","LLY"
    ]},

    # Consumer Staples
    **{t: "ConsumerStaples" for t in [
        "WMT","PG","COST","PEP","KO","KMB","CL","GIS",
        "HSY","KHC","KR","WBA","CVS","STZ","MO","PM",
        "BF-B","BTI","CAG","CPB","HRL","MKC","SJM","TAP","EL"
    ]},

    # Media / Telecom
    **{t: "MediaTelecom" for t in [
        "DIS","PARA","WBD","FOX","FOXA","NWSA","NWS",
        "CHTR","TMUS","VZ","T"
    ]},

    # Industrials
    **{t: "Industrials" for t in [
        "GE","CAT","HON","RTX","BA","DE","UPS","LMT","AXON"
    ]},

    # Energy
    **{t: "Energy" for t in [
        "XOM","COP","HAL","SLB","BKR","EOG","MPC",
        "PSX","VLO","OXY","DVN","APA","FANG","HES","KMI",
        "WMB","OKE","TRGP","ENB","EPD","MRO"
    ]},

    # Utilities
    **{t: "Utilities" for t in [
        "D","AEP","SO","DUK","EXC","XEL","ED","PEG","EIX",
        "NEE","PCG","SRE","DTE","WEC","PPL","ES","FE","AES","NRG"
    ]},

    # Real Estate
    **{t: "RealEstate" for t in [
        "PSA","PLD","AMT","CCI","EQIX","DLR","SBAC","SPG","O","VICI",
        "EQR","AVB","UDR","ESS","MAA","INVH","KIM","REG","FRT","BXP"
    ]},

    # Materials / Chemicals / Metals / Packaging
    **{t: "Materials" for t in [
        "LIN","NEM","FCX","NUE","STLD","X","AA","CMC","RS","ATI",
        "PKG","IP","WRK","BALL","SEE","SON",
        "CF","MOS","NTR","LYB","DOW","PPG","APD","ECL","SHW","ALB",
        "FMC","IFF","CE","DD","EMN"
    ]}
}

# Optional sanity checks
all_candidates = set(SP500_CANDIDATES)
all_mapped = set(SUBSECTOR_MAP.keys())

missing = sorted(all_candidates - all_mapped)
extra = sorted(all_mapped - all_candidates)

if __name__ == "__main__":
    print("Candidate count:", len(SP500_CANDIDATES))
    print("Unique candidate count:", len(all_candidates))
    print("Mapped count:", len(SUBSECTOR_MAP))
    print("Missing tickers:", missing)
    print("Extra tickers:", extra)
    print("All covered correctly:", len(missing) == 0 and len(extra) == 0)