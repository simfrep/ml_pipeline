import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scorecardpy as sc
import joblib
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve

from pathlib import Path

import random

def plots_1mw(
    results, 
    cl_models = None, 
    prob_models = None,
    ax=None,
    ix_dataset=1,
    descr_model='rf',
    price_open='EXAA',
    price_close='EPEX',
    naive_sign =1):
    
    results= results.drop(columns=results.filter(regex='xgb|naive|coin').columns)
    
    # Generic Benchmarks
    results['naive_sign'] = naive_sign
    results['coin_sign'] = np.random.choice([1,-1],results.shape[0])
    
    for key in cl_models.keys():
        results[key+'_results'] = cl_models[key][0].predict(cl_models[key][ix_dataset])
        results.loc[results[key+'_results'] == True,key+'_sign'] = 1
        results.loc[results[key+'_results'] == False,key+'_sign'] = -1
        
    for key in prob_models.keys():
        results[key+'_results'] =  prob_models[key][0].predict(prob_models[key][ix_dataset])
        results.loc[results[key+'_results'] <= 0.5,key+'_sign'] = -1
        results.loc[results[key+'_results'] >= 0.5,key+'_sign'] = 1
    
    for key in list(cl_models.keys())+list(prob_models.keys())+['naive','coin']:        
        results[key+"_profit"] = (results[price_close]-results[price_open])*(results[key+'_sign']) 
        results[key+"_profit_cum"] = results[key+"_profit"].cumsum()   
    
    for m in list(cl_models.keys())+list(prob_models.keys())+['naive','coin']:
        profvar=m+'_profit'
        signvar=m+'_sign'
        txt_profitabs = np.round(results[profvar].sum())
        txt_profitpct = round(100*results[profvar].sum()/abs(results[price_open]-results[price_close]).sum(),2)
        txt_profitmax = round(abs(results[price_open]-results[price_close]).sum(),2)
        txt_profitmwh = round(results[profvar].sum()/results[signvar].abs().sum(),2)
        f_trade = results[signvar] != 0
        f_correct = results[signvar]==np.sign(results.spread_sign)
        txt_corspread = round(100*len(results.loc[f_trade & f_correct])/len(results.loc[(f_trade)]),1)
        text = "{} Profit: {} EUR ({}% of max. {} EUR) Profit/MWh [€]: {} Accuracy: {}%".format(
        m,
        txt_profitabs,
        txt_profitpct,
        txt_profitmax,
        txt_profitmwh,txt_corspread)
    
        ax.plot(results.index,results[m+'_profit_cum'], label = text)
    
    plt.legend(loc=2)

    
def plot_model_analysis(results, model=None,dpred=None,mname=None,axes=None,steps=None,price_open='EXAA',price_close='EPEX',naive_sign=1):
    
    results= results.drop(columns=results.filter(regex='xgb|naive|coin').columns)
    
    # Generic Benchmarks
    results['naive_sign'] = naive_sign
    results['coin_sign'] = np.random.choice([1,-1],results.shape[0])

    cutoff = 0.01
    for f in steps:
        m = mname+'_'+str(f)
        results[m+'_results'] =  model.predict_proba(dpred)[:,1]
        results[m+'_sign'] = 0
        results.loc[results[m+'_results'] <= 0.5-f*cutoff,m+'_sign'] = -1
        results.loc[results[m+'_results'] >= 0.5+f*cutoff,m+'_sign'] = 1
        results[m+"_profit"] = (results[price_close]-results[price_open])*(results[m+'_sign'])  
        results[m+"_profit_cum"] = results[m+"_profit"].cumsum()

    for var in ['naive','coin']:
        results[var+"_profit"] = (results[price_close]-results[price_open])*(results[var+'_sign'])  
        results[var+"_profit_cum"] = results[var+"_profit"].cumsum()        
    
    # Plot Prob. Histogram
    results[mname+'_0_results'].hist(bins=100,ax=axes[0],range=(0,1))
    
    
    plot_confusion_matrix(model, dpred, results['target'],ax=axes[1],normalize='true')
    plot_precision_recall_curve(model, dpred, results['target'],ax=axes[2])
    plot_roc_curve(model, dpred, results['target'],ax=axes[3])
    #results[mname+'_0_results'].hist(bins=100)
    # Plot cum. results
    # Plot cum. results
    for m in [mname+'_'+str(s) for s in steps]+['naive','coin']:
        profvar=m+'_profit'
        signvar=m+'_sign'
        txt_profitabs = np.round(results[profvar].sum())
        txt_profitpct = round(100*results[profvar].sum()/abs(results[price_open]-results[price_close]).sum(),2)
        txt_profitmax = round(abs(results[price_open]-results[price_close]).sum(),2)
        txt_profitmwh = round(results[profvar].sum()/results[signvar].abs().sum(),2)
        f_trade = results[signvar] != 0
        f_correct = results[signvar]==np.sign(results.spread_sign)
        txt_corspread = round(100*len(results.loc[f_trade & f_correct])/len(results.loc[(f_trade)]),1)
        text = "{} Profit: {} EUR ({}% of max. {} EUR) Profit/MWh [€]: {} Accuracy: {}%".format(
        m,
        txt_profitabs,
        txt_profitpct,
        txt_profitmax,
        txt_profitmwh,txt_corspread)
        
        axes[4].plot(results.index,results[m+'_profit_cum'], label = text)
        axes[4].set_title('Variation: No position if model is undecided, i.e model_<x> is flat if p in [.5-x,-5+x]')
        
    plt.grid()
    plt.legend(loc=2)
    
    
def plot_cutoffshift(results, model=None,dpred=None,mname=None,ax=None,steps=None,price_open='EXAA',price_close='EPEX',naive_sign=1):
    
    results= results.drop(columns=results.filter(regex='xgb|naive|coin').columns)
    
    # Generic Benchmarks
    results['naive_sign'] = naive_sign
    results['coin_sign'] = np.random.choice([1,-1],results.shape[0])

    
    for f in steps:
        cutoff = 0.5 + f * 0.01
        m = mname+'_'+str(f)
        results[m+'_results'] =  model.predict_proba(dpred)[:,1]
        results[m+'_sign'] = 0
        
        results.loc[results[m+'_results'] <= cutoff,m+'_sign'] = -1
        results.loc[results[m+'_results'] >= cutoff,m+'_sign'] = 1
        results[m+"_profit"] = (results[price_close]-results[price_open])*(results[m+'_sign'])  
        results[m+"_profit_cum"] = results[m+"_profit"].cumsum()

    for var in ['naive','coin']:
        results[var+"_profit"] = (results[price_close]-results[price_open])*(results[var+'_sign'])  
        results[var+"_profit_cum"] = results[var+"_profit"].cumsum()        
    
    # Plot cum. results
    for m in [mname+'_'+str(s) for s in steps]+['naive','coin']:
        profvar=m+'_profit'
        signvar=m+'_sign'
        txt_profitabs = np.round(results[profvar].sum())
        txt_profitpct = round(100*results[profvar].sum()/abs(results[price_open]-results[price_close]).sum(),2)
        txt_profitmax = round(abs(results[price_open]-results[price_close]).sum(),2)
        txt_profitmwh = round(results[profvar].sum()/results[signvar].abs().sum(),2)
        f_trade = results[signvar] != 0
        f_correct = results[signvar]==np.sign(results.spread_sign)
        txt_corspread = round(100*len(results.loc[f_trade & f_correct])/len(results.loc[(f_trade)]),1)
        text = "{} Profit: {} EUR ({}% of max. {} EUR) Profit/MWh [€]: {} Accuracy: {}%".format(
        m,
        txt_profitabs,
        txt_profitpct,
        txt_profitmax,
        txt_profitmwh,txt_corspread)
        
        
        ax.plot(results.index,results[m+'_profit_cum'], label = text)
        # Show Grid and Legend
    plt.grid() 
    plt.legend(loc=2)    
    
def res_plot_w_vols(results, model=None, dpred=None,mname=None,ax0=None,ax1=None,cut_bins=None,cut_labels=None,benchmarkvol=1, cutoff=0,price_open='EXAA',price_close='EPEX'):
    
    results= results.drop(columns=results.filter(regex='xgb|naive|coin').columns)

    for m in [mname,'baseline']:

        results[m+'_results'] =  model.predict_proba(dpred)[:,1]
        results[m+'_volume'] = pd.cut(results[m+'_results'],bins=cut_bins,labels=cut_labels,right=False).astype(float)
        if m == 'baseline':
            results[m+'_volume'] = 0
            results.loc[results[m+'_results'] <= 0.5-cutoff,m+'_volume'] = pd.cut(results[m+'_results'],bins=cut_bins,labels=cut_labels,right=False).astype(float)
            results.loc[results[m+'_results'] >= 0.5+cutoff,m+'_volume'] = pd.cut(results[m+'_results'],bins=cut_bins,labels=cut_labels,right=False).astype(float)
        
        results[m+'_benchmark_volume'] = np.sign(results[m+'_volume']) * benchmarkvol
        
    for m in [mname,'baseline',mname+'_benchmark', 'baseline_benchmark']:   
        results[m+"_profit"] = (results[price_close]-results[price_open])*(results[m+'_volume'])  
        results[m+"_profit_cum"] = results[m+"_profit"].cumsum()      
    
    # Plot Prob. Histogram
    results[mname+'_results'].hist(bins=50,ax=ax0)
    # Plot cum. results
    results[[mname+'_profit_cum','baseline_profit_cum',mname+'_benchmark_profit_cum', 'baseline_benchmark_profit_cum']].plot(grid=True,ax=ax1)
    
    plt.legend(loc=2)
    print("max. achievable profit", np.round(abs(results[m+"_profit"]).sum()),"EUR")
    profvar='xgb_profit'
    signvar='xgb_volume'
    text1 = "Profit:"+str(np.round(results[profvar].sum()))+"EUR ("+str(round(100*results[profvar].sum()/results[profvar].abs().sum(),2))+"% of max Profit) Profit/MWh [€]: "+ str(round(results[profvar].sum()/results[signvar].abs().sum(),2))+" sprea " +str(round(100*len(results.loc[(results[signvar] != 0) &(np.sign(results[signvar])==np.sign(results.spread_sign))])/len(results.loc[(results[signvar] != 0)]),1))
    ax1.set_title(text1)        
 
    return results