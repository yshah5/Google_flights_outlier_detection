#!/usr/bin/python
# coding=utf-8

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import pandas as pd
import datetime
from dateutil.parser import parse
import time
import os
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean


import plotly.plotly as py
import plotly.tools as tls

def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]

def strip_accents(text):
    # function for removing accents from name of places or location
    try:
        text = unicode(text, 'utf-8')
    except: # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def quartiles(dataPoints):
      # check the input is not empty
      if not dataPoints:
        raise 'no data points passed'
       # 1. order the data set
      sortedPoints = sorted(dataPoints)
       # 2. divide the data set in two halves
      mid = len(sortedPoints) / 2
      if (len(sortedPoints) % 2 == 0):
           # even
           lowerQ = median(sortedPoints[:mid])
           upperQ = median(sortedPoints[mid:])
      else:
           # odd
           lowerQ = median(sortedPoints[:mid])  # same as even
           upperQ = median(sortedPoints[mid + 1:])
      return (lowerQ, upperQ)

def scrape_data(start_date, from_place, to_place, city_name):
    
    # function to extract flight date and price for 60 days from given day
    
    flight_details = []
    final_flight_details = []



    chromedriver = "/Users/yash/Downloads/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(chromedriver)


    driver.get('https://www.google.com/flights/explore/')
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]/div/div')
    from_input.click()

    actions = ActionChains(driver)
    actions.send_keys(from_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()

    time.sleep(0.1)

    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()

    actions = ActionChains(driver)
    actions.send_keys(to_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(0.9)


    temp_url = driver.current_url.split('d=')[0]


    new_url = temp_url + 'd=' + str(start_date.year) + '-0' + str(start_date.month) + '-' + str(start_date.day)

    driver.close()
    newdriver=webdriver.Chrome(chromedriver)
    newdriver.get(new_url)
    time.sleep(0.9)



    results = newdriver.find_elements_by_class_name('LJTSM3-v-d')
    time.sleep(0.9)
    i=0

    for result in results:

        city = strip_accents(result.find_element_by_class_name('LJTSM3-v-c').text)
        time.sleep(0.9)

        cityname=city.split(',')[0].strip().lower()

        if cityname == city_name.lower():
            bars = result.find_elements_by_class_name('LJTSM3-w-x')
            time.sleep(0.9)
            for bar in bars:

                ActionChains(newdriver).move_to_element(bar).perform()
                time.sleep(0.9)


                flight_details.append(
                    (result.find_element_by_class_name('LJTSM3-w-w').text,
                     result.find_element_by_class_name('LJTSM3-w-h').text))
                i=i+1

            break
        else:
            continue

    for fd in flight_details:
        price = float(fd[0].replace('$', '').replace(',', ''))
        date = parse(fd[1].split('-')[0].strip())
        final_flight_details.append((price, date))

    df = pd.DataFrame(final_flight_details, columns=['Price', 'Start-Date'])
    print df

def scrape_data_90(start_date, from_place, to_place, city_name):
    
    # function to extract the flight date and price for 90 days from the given day.
    
    flight_details = []
    final_flight_details = []

    chromedriver = "/Users/yash/Downloads/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(chromedriver)

    driver.get('https://www.google.com/flights/explore/')
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]/div/div')
    from_input.click()

    actions = ActionChains(driver)
    actions.send_keys(from_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()

    time.sleep(0.1)

    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()

    actions = ActionChains(driver)
    actions.send_keys(to_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(0.9)


    temp_url = driver.current_url.split('d=')[0]


    new_url = temp_url + 'd=' + str(start_date.year) + '-0' + str(start_date.month) + '-' + str(start_date.day)

    driver.close()
    newdriver = webdriver.Chrome(chromedriver)
    newdriver.get(new_url)
    time.sleep(0.9)

    results = newdriver.find_elements_by_class_name('LJTSM3-v-d')
    time.sleep(0.9)
    i = 0

    for result in results:

        city = strip_accents(result.find_element_by_class_name('LJTSM3-v-c').text)
        time.sleep(0.9)

        cityname = city.split(',')[0].strip().lower()

        if cityname == city_name.lower():
            bars = result.find_elements_by_class_name('LJTSM3-w-x')
            time.sleep(0.9)
            for bar in bars:
                ActionChains(newdriver).move_to_element(bar).perform()
                time.sleep(0.9)

                flight_details.append(
                    (result.find_element_by_class_name('LJTSM3-w-w').text,
                     result.find_element_by_class_name('LJTSM3-w-h').text))
                i = i + 1

            break
        else:

            continue

    newdriver.find_element_by_class_name('LJTSM3-w-C').click()

    time.sleep(0.9)
    results = newdriver.find_elements_by_class_name('LJTSM3-v-d')
    time.sleep(0.9)


    for result in results:

        city = strip_accents(result.find_element_by_class_name('LJTSM3-v-c').text)
        time.sleep(0.9)

        cityname = city.split(',')[0].strip().lower()

        if cityname == city_name.lower():
            bars = result.find_elements_by_class_name('LJTSM3-w-x')
            time.sleep(0.9)
            for bar in bars:
                if i==90:
                    break

                else:
                    ActionChains(newdriver).move_to_element(bar).perform()
                    time.sleep(0.9)
                    if (result.find_element_by_class_name('LJTSM3-w-w').text,result.find_element_by_class_name('LJTSM3-w-h').text) in flight_details:
                        continue
                    else:
                        flight_details.append(
                            (result.find_element_by_class_name('LJTSM3-w-w').text,
                            result.find_element_by_class_name('LJTSM3-w-h').text))
                        i=i+1

            break
        else:
            continue

    for fd in flight_details:

        price = float(fd[0].replace('$', '').replace(',', ''))
        date = parse(fd[1].split('-')[0].strip())
        final_flight_details.append((price, date))

    flight_datalist=final_flight_details
    df = pd.DataFrame(final_flight_details, columns=['Price', 'Start-Date'])
    task_3_dbscan(df)
    task_3_IQR(df)

def task_3_dbscan(flight_data):
    
    # function to perform DBSCAN clustering and identify outliers (outrageous flight price) 
    price = np.array([x for x in flight_data['Price']])
    final_noise_point = []
    outliers = []

    days = np.arange(90)

    price_df = pd.DataFrame(price, columns=['Price']).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(np.arange(len(flight_data['Price'])), flight_data['Price'])
    price_s = MinMaxScaler(feature_range=(-2.5, 2.5)).fit_transform(price[:, None])
    days_s = MinMaxScaler(feature_range=(-2, 2)).fit_transform(days[:, None])
    X = np.concatenate([days[:, None], price[:, None]], axis=1)

    X_ss = np.concatenate([days_s, price_s], axis=1)

    db = DBSCAN(eps=0.45, min_samples=5).fit(X_ss)

    labels = db.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.subplots(figsize=(12, 8))

    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X_ss[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    plt.title("Total Clusters: {}".format(clusters), fontsize=14,
              y=1.01)
    pf = pd.concat([price_df, pd.DataFrame(db.labels_,
                                           columns=['cluster'])], axis=1)

    lbls = np.unique(db.labels_)
    print "Cluster labels: {}".format(np.unique(lbls))

    cluster_means = [np.mean(X_ss[labels == num, :], axis=0) for num in range(lbls[-1] + 1)]
    print "Cluster Means: {}".format(cluster_means)

    noise_points = X_ss[labels == -1]

    for noise_point in noise_points:
        index_of_nearest_cluster = []
        dist = []
        noise_point_price = X[X_ss.tolist().index(np.array(noise_point).tolist())][1]
        noise_point_day = X[X_ss.tolist().index(np.array(noise_point).tolist())][0]


        for num in range(lbls[-1] + 1):
            dist.append([num, euclidean(noise_point, cluster_means[num])])

        nearest_cluster = min(dist, key=lambda x: x[1])[0]

        nearest_cluster_values = X_ss[labels == nearest_cluster]
        for cluster_point in nearest_cluster_values:
            point = np.array(cluster_point).tolist()

            index_of_nearest_cluster.append(X_ss.tolist().index(point))

        nearest_price = [X[i][1] for i in index_of_nearest_cluster]
        std_2 = np.std(nearest_price)*2
        mean_cluster = np.mean(nearest_price)
        if abs((noise_point_price - mean_cluster)) > std_2 and abs((noise_point_price - mean_cluster)) >= 50:
            final_noise_point.append([noise_point_price, noise_point_day])

    for final_point in final_noise_point:
        start = int(final_point[1])
        end = int(final_point[1] + 1)
        final_price = flight_data[start:end]['Price'].values.tolist()
        final_date = str(flight_data[start:end]['Start-Date'].values).split('T')[0].replace("['","")
        outliers.append([final_price[0], final_date])

    outliers_df = pd.DataFrame(outliers, columns=['Price', 'Start-Date'])
    print outliers_df
    plt.show(block=True)

def task_3_IQR(flight_data):
    
    # function to detect outliers using Interquartile range
    
    outliers=[]
    outliers_price=[]
    Prices=flight_data['Price'].values.tolist()
    fquart,tquart=quartiles(Prices)

    IQR=tquart-fquart

    llimit=fquart-(1.5*IQR)

    ulimit=tquart+(1.5*IQR)

    for Price in Prices:
        if llimit<=Price<=ulimit:
            continue
        else:
            outliers_price.append(Price)
    for outlier in outliers_price:

        final_price = flight_data[Prices.index(outlier):(Prices.index(outlier)+1)]['Price'].values.tolist()
        final_date = str(flight_data[Prices.index(outlier):(Prices.index(outlier)+1)]['Start-Date'].values).split('T')[0].replace("['", "")
        outliers.append([final_price[0], final_date])
    outliers_df = pd.DataFrame(outliers, columns=['Price', 'Start-Date'])
    print outliers_df
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(Prices)
    plt.show(block=True)

datetime_object = datetime.strptime('apr 28 2017', '%b %d %Y')
scrape_data_90(datetime_object, "New York", "Norway", "Alesund")

