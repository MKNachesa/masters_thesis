import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import os
from selenium import webdriver

thesis = "C:/Users/mayan/Documents/Language Technology Uppsala/Thesis"
data_path = os.path.join(thesis, "metadata/data")
riksdagen = os.path.join(data_path, "riksdagen_speeches.parquet")

def add_politician_age_at_debatedate():
    speakers = df["text"].lower().to_list()
    print("Reading webpage")
    URL = "https://riksdagen.se"
    page = requests.get(URL+"/sv/ledamoter-partier/")
    soup = bs(page.content, "html.parser")

    # initial method for getting "extra" titles (at the start)
##    titles = set()
##    for member in list(filter(lambda x: len(x.split())>=4, members)):
##        titles.update(member.split()[:-3])

    # manually filtered titles
    titles = {'landsbygdsminister', 'försvarsmin.', 'utbildningsminister',
              'och', 'miljöminister', 'tredje', 'förste', 'justitieminister',
              'etableringsmin.', 'socialminister', 'demokratiminister',
              'statsrådet', 'klimatminister', 'talman', 'närings-', 'vice',
              'näringsminister', 'finansminister', 'klimat-', 'kulturminister',
              'försvarsminister', 'arbetsm.-', 'utrikesminister',
              'idrottsminister', 'arbetsmarknadsminister', 'energiminister',
              'jordbruksminister', 'arbetsmarknads-', 'kultur-',
              'jämställdhetsminister', 'statsminister', 'arbetsmarknadsmin.',
              'infrastruktur-', 'miljö-'}

    proper_names = []
    for i, row in df.iterrows():
        name = row["text_lower"]
        if name == "walburga habsburg dougla (m)":
            name = "walburga habsburg douglas (m)"
        elif name == "cecilia wikström i uppsa (fp)":
            name = "cecilia wikström (fp)"
        name = " ".join(list(filter(lambda x: x not in titles, name.split()))[:-1])
        proper_names.append(name)

    proper_names = list(set(proper_names))
    
    members = set(df["text_lower"].to_list())

    mydivs = soup.find_all("div", {"class": "fellows-group"})

    member_urls = [member.a["href"] for i in range(26)
                   for member in mydivs[i].ul.find_all("li")]

    fullname_to_shortname = dict()
    member_info = dict()

    total_members = len(member_urls)

    for i, member_url in enumerate(member_urls):
        page = requests.get(URL+member_url)
        soup = bs(page.content, "html.parser")
        name = soup.find_all("h1", {"class": "biggest fellow-name"})[0].get_text()
        name = " ".join(name.split()[:-1]).lower()
        info = soup.find_all("div",
                             {"class":
                              "large-12 medium-12 small-12 columns fellow-item"})
        info = list(map(lambda x: x.get_text(), info))
        info = list(filter(lambda x: "Född" in x, info))[0]
        year = int(info.strip().split()[-1])
        member_info[name] = year

        to_remove = []
        for fullname in members:
            if name in fullname:
                fullname_to_shortname[fullname] = name
                to_remove.append(fullname)
        for to_r in to_remove:
            members.remove(to_r)

        print(f"Processed {i:>3} out of {total_members} members")
    ##    break
        
    return fullname_to_shortname, member_info


if __name__ == "__main__":
    print("Opening parquet")
    df = pd.read_parquet(riksdagen)
    df["text_lower"] = df["text"].apply(lambda x: x.lower())

##    driver = webdriver.Chrome()
##    driver.get("https://www.google.com/")
##    l = driver.find_element("id", "L2AGLb")
##    l.click()
