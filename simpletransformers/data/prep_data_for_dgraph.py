import os
import re
import json
import pandas as pd
from collections import defaultdict

import sys
sys.path.insert(0, '../../')
from simpletransformers.data.data_utils import s3, BUCKET, get_json
from simpletransformers.data.url_vocab import UrlVocab


if __name__ == '__main__':
    # Load URL metadata
    url_file_path = os.path.join('jie-faq', 'faq', 'outreach_support_url_meta_addHrefAndNav.json')
    jsonstring = get_json(s3, BUCKET, url_file_path)
    df_url = pd.io.json.json_normalize(jsonstring)

    # Remove some URLs from the predication candidates
    df_url = df_url[(~df_url['header'].isnull()) & (~df_url['article'].isnull())]
    df_url = df_url[df_url['header'].apply(lambda x: len(x) > 0) & df_url['article'].apply(lambda x: len(x) > 0)]
    df_url = df_url[df_url['nav_hrefs'].apply(lambda x: len(x) > 0)]

    # Remove surrounding white spaces in text fields
    df_url['header'] = df_url['header'].apply(lambda x: x.strip())


    def parse(url):
        if url.startswith('/hc/en-us/categories/'):
            match = re.match(r'/hc/en-us/categories/(\d+)(?:-(.*))?', url)
            assert match is not None
            mid, name = match.groups()
            return 'category', mid, name
        elif url.startswith('/hc/en-us/sections'):
            match = re.match(r'/hc/en-us/sections/(\d+)(?:-(.*))?', url)
            assert match is not None
            mid, name = match.groups()
            return 'section', mid, name
        elif url.startswith('https://support.outreach.io/hc/en-us/articles'):
            match = re.match(r'https://support.outreach.io/hc/en-us/articles/(\d+)(?:-(.*))?', url)
            assert match is not None
            mid, name = match.groups()
            return 'article', mid, name
        elif url == '/hc/en-us':
            return 'root', '0', 'root'
        else:
            raise ValueError(f'Can not parse URL: {url}')


    url_to_mid = {}

    # Prepare navigation URL data
    nav_urls =  sorted(list(set(sum(df_url['nav_hrefs'], []))))
    df_nav_url = pd.DataFrame([parse(x) for x in nav_urls], columns=['type', 'mid', 'nav_url_names'])
    df_nav_url['url'] = nav_urls

    # Prepare article URL data
    urls = list(df_url['url'])
    df_article_url = pd.DataFrame([parse(x) for x in urls], columns=['type', 'mid', 'none'])
    df_article_url['url'] = urls
    df_article_url['header'] = df_url['header']
    df_article_url['article'] = df_url['article']
    df_article_url.drop(columns=['none'])

    # Verifying that mid is unique
    assert len(df_nav_url) == len(df_nav_url.mid.unique())
    assert len(df_article_url) == len(df_article_url.mid.unique())
    assert len(set(df_article_url.mid) & set(df_nav_url.mid)) == 0

    for row in df_nav_url.to_dict(orient='records'):
        assert row['url'] not in url_to_mid
        url_to_mid[row['url']] = row['mid']
    for row in df_article_url.to_dict(orient='records'):
        assert row['url'] not in url_to_mid
        url_to_mid[row['url']] = row['mid']

    # Add cross-article links
    def hrefs_to_mids(hrefs, url_to_mid):
        mids = []
        for h in hrefs:
            if h in url_to_mid:
                mids.append(url_to_mid[h])
            else:
                print(f'URL {h} does not have a mid.')
        return mids

    df_article_url['hrefs'] = df_url['hrefs'].apply(lambda x: hrefs_to_mids(x, url_to_mid))
    df_article_url['nav_hrefs'] = df_url['nav_hrefs'].apply(lambda x: hrefs_to_mids(x, url_to_mid))

    # Write RDF to file
    f = open('./support_url.rdf', 'w')
    back_edge = defaultdict(set)
    for row in df_nav_url.to_dict(orient='records'):
        # Relations for root/category/section nodes
        f.write(f'_:{row["mid"]} <type> \"{row["type"]}\" .\n')
        f.write(f'_:{row["mid"]} <url> \"{row["url"]}\" .\n')
        f.write(f'_:{row["mid"]} <name> \"{row["nav_url_names"]}\" .\n')
        # f.write('\n')
    for row in df_article_url.to_dict(orient='records'):
        # Relations for article nodes
        f.write(f'_:{row["mid"]} <type> \"{row["type"]}\" .\n')
        f.write(f'_:{row["mid"]} <url> \"{row["url"]}\" .\n')
        f.write(f'_:{row["mid"]} <header> {json.dumps(row["header"])} .\n')
        f.write(f'_:{row["mid"]} <article> {json.dumps(row["article"])} .\n')
        # Hyperlink relation for article nodes
        for href_mid in row["hrefs"]:
            f.write(f'_:{row["mid"]} <href> _:{href_mid} .\n')
        # Categorical relation for article nodes, tracing back to root
        if row["nav_hrefs"]:
            nav_hrefs = row["nav_hrefs"][::-1]
            f.write(f'_:{row["mid"]} <sub> _:{nav_hrefs[0]} .\n')
            for i in range(len(nav_hrefs) - 1):
                if nav_hrefs[i+1] not in back_edge[nav_hrefs[i]]:
                    back_edge[nav_hrefs[i]] = nav_hrefs[i+1]
                    f.write(f'_:{nav_hrefs[i]} <sub> _:{nav_hrefs[i+1]} .\n')
        # f.write('\n')

    f.close()