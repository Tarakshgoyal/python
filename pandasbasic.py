import numpy as np
import pandas as pd
dict={
    "name":['harry','rohan','skillf','shubh'],
    "marks":[92,34,24,17],
    "city":['rampur','kolkata','bareilly','antarctica']
}
df=pd.DataFrame(dict) #converts to excel sheet
df.to_excel("output.xlsx",index=False)#creates an excel file 
df.to_csv("out.csv",index=False)
print(df.head(2))#print 1st 2 data
print(df.tail(2))#print last 2 data
print(df.describe())#statist analysis of numric value 
t=pd.read_csv('out.csv')#read csv
print(t)
t['name'][0]='taraksh'
print(t)
t.to_csv('out.csv')#apply changes
print(t)
t.index=['first','second','third','forth'] #added a custom index
print(t)
ser=pd.Series(np.random.rand(34))
type(ser)
print(ser)
newdf=pd.DataFrame(np.random.rand(334,5),index=np.arange(334))
type(newdf)
print(newdf.head())
newdf.describe()
print(newdf)
newdf[0][0]='taraksh'
print(newdf.dtypes)
print(newdf.head())
print(newdf.index)
print(newdf.to_numpy())
newdf.sort_index(axis=0, ascending=False)
print(newdf.head())
type(newdf[0])
newdf1=newdf #pointer newdf1 will store the address
newdf2=newdf.copy()#just values are stored
newdf.loc[0,0]=654 #changes without warning
print(newdf.head())
newdf.columns=list("ABCDE")
print(newdf.head())
newdf.loc[0,0]=654#will create new column 
print(newdf.head(2))
newdf.loc[0,'A']=654975
print(newdf.head())
newdf=newdf.drop(0,axis=1)#to remove a column
print(newdf.head())
print(newdf.loc[[1,2],['C','D']])#printf selected columns
newdf.loc[(newdf['A']<0.3) & (newdf['C']>0.1)]
print(newdf.loc[(newdf['A']<0.3) & (newdf['C']>0.1)])
newdf.iloc[0,4]
print(newdf.iloc[0,4])
newdf.drop(['A','D'],axis=1,inplace=True)
print(newdf)
newdf.drop([1,5],axis=0,inplace=True)
print(newdf.head(3))
newdf.reset_index(drop=True,inplace=True)
print(newdf.head(3))
newdf.loc[:,['B']]=56
print(newdf['B'].isnull())
print("Datafram has been saved ")
df.info() #to know number of indexes
df.shape #to know number of indexes
df.sort_values(by=['marks'], ascending=False)
df["Value"].mode()
df.query("marks==92").sort_values(by="Char Count")