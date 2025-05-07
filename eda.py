import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df1 = pd.read_csv("test.csv")
df2 = pd.read_csv("train.csv")
df = pd.concat([df1,df2],axis=0)
pd.set_option("display.max_columns",None)
# print(df.loc[nan_row[0]])
df.dropna(inplace=True) #simply dropping the unwanted row made our data clean
# print(df.isna().sum())

print("\n\n")
nan_row = df.loc[df.isna().any(axis=1) == True].index.to_list() #finding index of rows with nan values. there is a single row where all values are nan

#I have performed EDA on all the parameters but i will show the outputs where there is an unusual trend or pattern.

#Data Processing
# print(df["Age"].max(),df["Age"].min()) 65 and 18
bins_age = [17,25,35,45,55,65] #it works as (a,b]
labels_age = ["Young Adults","Adults","Matured Adults","Middle Age","Senior"]
df["Age_Label"] = pd.cut(df["Age"],bins=bins_age,labels=labels_age)
# print(df["Tenure"].max(),df["Tenure"].min()) 60 and 1
bins_tenure = [0,6,24,36,48,60]
labels_tenure = ["New","Hooked","Regular","Long-Term","Loyal"]
df["Tenure_Label"] = pd.cut(df["Tenure"],bins=bins_tenure,labels=labels_tenure)
support_labels = ["0-3","3-6","6-10"]
support_bins = [-1,3,6,10]
df["SupportCallsRange"] = pd.cut(df["Support Calls"],labels=support_labels,bins=support_bins)

#Based on the labels, i will create a metric to find who churn more - churning_based_on_label/total_people_in_that_label
def create_metric(column,name):
    churn = df.loc[df["Churn"]==1][column].value_counts().reset_index().sort_values(column)["count"].to_numpy()
    total = df[column].value_counts().reset_index().sort_values(column)["count"].to_numpy(dtype=float)
    metric = churn/total
    # metric2 = no_churn/total
    return pd.DataFrame(data={
        name:df.loc[df["Churn"]==1][column].value_counts().reset_index().sort_values(column)[column],
        "Churn Rate":metric,
    })

age_metric = create_metric("Age_Label","Age")
tenure_metric = create_metric("Tenure_Label","Tenure")
subs_metric = create_metric("Subscription Type","Subscription")
churn_by_calls = df.groupby("Support Calls",observed=False)["Churn"].mean().reset_index()
churn_by_age_gender = df.groupby(["Age_Label","Gender"],observed=False)["Churn"].mean().reset_index()
churn_by_age_gender = churn_by_age_gender.pivot(index="Age_Label",
                               columns="Gender",
                               values="Churn"
                               )


plt.subplot(2,2,1)# plt.title("Churn vs No Churn")
plt.title("Amount of Churn vs No Churns")
plt.pie(df["Churn"].value_counts(),labels=["Churn","No Churn"],autopct="%.2f %%",shadow=True)
plt.subplot(2,2,2)
plt.plot(churn_by_calls["Support Calls"],churn_by_calls["Churn"],marker="o",linestyle="-")
plt.xlabel("Support Calls")
plt.ylabel("Mean Churn")
plt.title("Support Calls Vs Churn")
plt.xticks(np.arange(start=1,stop=11))
plt.yticks([0.25,0.5,1])
plt.subplot(2,2,3)
x = list(range(len(churn_by_age_gender)))
width = 0.35
plt.bar([i + width/2 for i in x],churn_by_age_gender["Male"],width=width,label="Male",color="#219aa3")
plt.bar([i - width/2 for i in x],churn_by_age_gender["Female"],width=width,label="Female",color="#e627d2")
plt.xticks(ticks=x,labels=churn_by_age_gender.index)
plt.ylabel("Mean Churn")
plt.title("Variation of Mean Churn With Different People")
plt.legend()
plt.tight_layout()
print(df.columns)
print(df.head())
# print(df.groupby("Contract Length")["Churn"].mean())
plt.subplot(2,2,4)
churn_by_delay = df.groupby("Payment Delay",observed=False)["Churn"].mean().reset_index()
plt.plot(churn_by_delay["Payment Delay"],churn_by_delay["Churn"],marker="o",linestyle="-")
plt.xlabel("Payment Delay")
plt.ylabel("Mean Churn")
plt.title("Payment Delay Vs Churn")
plt.xticks(np.arange(start=1,stop=31))
plt.yticks([0.25,0.5,1])
plt.show()
del x,width
# # churn = df.loc[df["Churn"]==1]
# no_churn = df.loc[df["Churn"]==0]