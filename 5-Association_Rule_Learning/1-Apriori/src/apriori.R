# Apriori Associated Rule Learning

# Path to the file
BinPath = paste(getwd(), 'MachineLearning-Portfolio/5-Association_Rule_Learning/1-Apriori/bin', sep='/')
FilePath = paste(BinPath, '/Market_Basket_Optimisation.csv', sep='')
setwd(BinPath)

# Importing Libraries
#install.packages('caTools', repos = "http://cran.us.r-project.org",dependencies = TRUE)
#install.packages('arules',repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages('arulesViz',repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(caTools, help, pos = 2, lib.loc = NULL)
library(arules)
library(arulesViz)

# Import dataset
dataset = read.csv(FilePath, header = FALSE)
dataset = read.transactions(FilePath, sep=",", rm.duplicates = TRUE)
summary(dataset)

# Getting the top frequently bought items
png("Top100_R.png") 
itemFrequencyPlot(dataset, topN = 100)
dev.off() 

png("Top10_R.png")
itemFrequencyPlot(dataset, topN = 10)
dev.off() 

# Training Apriori on the dataset
rules = apriori(data = dataset,
                parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
jpeg("ParacoordList_R.jpeg")
plot(sort(rules, by = 'lift')[1:10], 
    method = "paracoord", 
    control = list(verbose = TRUE))
dev.off()