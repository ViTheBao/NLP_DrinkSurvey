#############################################
###### Team 8 Wine Survey Analysis ##########
#############################################

#libraries needed for data analysis
library(rvest)
library(tidyverse)
library(tidytext)
library(dplyr)
library(tidyr)
library(widyr)
library(scales)
library(ggplot2)
library(textdata)
library(stringr)
library(wordcloud)
library(wordcloud2)
library(reshape2)
library(forcats)
library(tm)
library(pdftools)
library(reshape2)
library(ggraph)
library(igraph)
library(textreadr)
library(plotly)
library(quanteda)
library(RColorBrewer)
library(quanteda.textmodels)
library(shiny)
library(shinydashboard)

### Loading Data to R ###
setwd("/Users/ecemdiren/Desktop/T8_WineAnalysis/Surveys")
survey <- list.files(path = "/Users/ecemdiren/Desktop/T8_WineAnalysis/Surveys")

#creating a loop to read add all the files in by line
survey_df <- data_frame()

for (i in 1:length(survey)) {
  text_place <- read_document(survey[i])
  df_two <- data_frame(survey = i, question = 1:length(text_place), text = text_place)
  survey_df <- rbind(survey_df, df_two)
}

#load stop_words to analyze dfs
data(stop_words)

#tidy data, with stop words
survey_tidy <- survey_df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

#counting most frequent words without stopwords
survey_freq <- survey_tidy %>%
  count(word, sort=T)

#bar chart of the most frequent tokens
freq_hist <- survey_freq %>%
  group_by(word) %>%
  filter(n>10) %>%
  top_n(10) %>%
  mutate(word=reorder(word, n)) %>%
  ggplot(aes(word, n))+
  geom_col()+
  xlab(NULL)+
  coord_flip()
print(freq_hist)


#####################################################
### Correlation Analysis Between Survey Questions ###
#####################################################

#Pairwise analysis between freq words in the questions
survey_cors <- survey_tidy %>%
  group_by(word) %>%
  filter(n() >= 10) %>%
  pairwise_cor(word, question, sort=TRUE) #check correlation on how many times a word is in the answer

#finding correlations in relation to a specific word
survey_cors %>%
  filter(item1 == "cuisine")

survey_cors <- data.frame(survey_cors)

### correlation histogram ###
survey_cors %>%
  filter(item1 %in% c("food", "wine", "cuisine", "dinner","logical","creative")) %>%
  group_by(item1) %>%
  top_n(5) %>%
  ungroup() %>%
  mutate(item2 = reorder(item2, correlation)) %>%
  ggplot(aes(item2, correlation)) +
  geom_bar(stat = "identity", fill ="blue", colour ="blue")+
  facet_wrap(~item1, scales = "free")+
  coord_flip()


### Correlation Network ###
#expand plot area to see full network
#item1 grouped
survey_cors %>%
  group_by(item1) %>%
  filter(correlation >0.2) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr")+
  geom_edge_link(aes(edge_alpha = correlation), show.legend=F)+ #shows links between
  geom_node_point(color = "purple", size=6)+ #similar to ngram semantic method/template
  geom_node_text(aes(label=name), repel=T)+
  theme_void()

#item2 grouped
survey_cors %>%
  group_by(item2) %>%
  filter(correlation >0.2) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr")+
  geom_edge_link(aes(edge_alpha = correlation), show.legend=F)+ #shows links between
  geom_node_point(color = "purple", size=6)+ #similar to ngram semantic method/template
  geom_node_text(aes(label=name), repel=T)+
  theme_void()

#######################################################
########### Bi-grams and TF-IDF ANALYSIS #############
#######################################################

#creating bigrams for the survey 
survey_bigrams <- survey_df %>%
  unnest_tokens(bigram, text, token = "ngrams", n =2) %>% #2 refers to bigram
  filter(!is.na(bigram))

#frequent bigrams
bigram_freq <- survey_bigrams %>%
  count(bigram, sort = TRUE) 

#separating stop words from bigrams
bigrams_sep <- survey_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

#filtering out stop words
bigrams_filt<- bigrams_sep %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

#count most common bigram
bigram_counts <- bigrams_filt %>%
  count(word1, word2, sort = TRUE)
#want to see the new bigrams
bigram_counts

### Bi-grams and TF-IDF ###

bigram_unite <- bigrams_filt %>%
  unite(bigram, word1, word2, sep=" ") #we need to unite what we split in the previous section

#applying tf-idf to bigrams for questions
bigram_tf_idf <- bigram_unite %>%
  count(question, bigram) %>%
  bind_tf_idf(bigram, question, n) %>%
  arrange(desc(tf_idf))

view(bigram_tf_idf)

### Bi-gram TF-IDF GRAPH ###
# looking at the graphical approach:
bigram_tf_idf %>%
  arrange(desc(tf_idf)) %>%
  group_by(question) %>%
  filter(n > 1) %>%
  top_n(10) %>%
  ungroup() %>%
  ggplot(aes(bigram, tf_idf, fill=question))+
  geom_col(show.legend=FALSE)+
  labs(x=NULL, y="tf-idf")+
  facet_wrap(~question, ncol=2, scales="free")+
  coord_flip()


##################################################
########### Naive Bayes on Questions #############
##################################################

setwd("C:/Documents and Settings/Nguye/Documents/HOMEWORK/MSBA/Term 2/Text Analytics NLP/Assignments/Team Video/survey pdf/")
mn <- list.files(path="C:/Documents and Settings/Nguye/Documents/HOMEWORK/MSBA/Term 2/Text Analytics NLP/Assignments/Team Video/survey pdf/")

#read all PDF files and put into a list
survey_pdf <- readPDF(control = list(text = "-layout"))
surveys     <- Corpus(URISource(mn), readerControl = list(reader = survey_pdf))

#we need to convert the VCorpus from the previous point to
#a regular corpus using the corpus() function.
msg.dfm <- dfm(corpus(surveys), tolower = TRUE) #generating document 
msg.dfm <- dfm_trim(msg.dfm, min_termfreq = 2, min_docfreq = 1)
msg.dfm <- dfm_weight(msg.dfm)
#looking at the DFM
head(msg.dfm)

#let's split the docs into training and testing data
msg.dfm.train<-msg.dfm[1:26,] # 60% of the data is training
msg.dfm.test<-msg.dfm[26:35,] # 20% of the data is testing

#building the Naive Bayes model:
NB_classifier <- textmodel_nb(msg.dfm.train, c(1,1,1,0,0,0,0,1,0,0,
                                               1,1,0,1,0,1,1,1,1,1,
                                               1,1,1,0,1,1))
NB_classifier
summary(NB_classifier)

#predicting on the whole dataset
pred <- predict(NB_classifier, msg.dfm) 
pred
#predicted five survey answers wrong out of 44


##############################
##### sentiment analysis #####
##############################

### afinn ###
afinn <- get_sentiments("afinn")
nrc <- get_sentiments("nrc")
bing <- get_sentiments("bing")

#binding sentiments together
sentiments <- bind_rows(mutate(afinn, lexicon ="afinn"),
                        mutate(nrc, lexicon = "nrc"),
                        mutate(bing, lexicon = "bing"))
#sorting for afinn
afinn <- sentiments%>%
  filter(lexicon =="afinn")

# mean value 
survey_afinn <- survey_tidy %>%
  inner_join(get_sentiments("afinn"))%>%
  summarise(mean(value))

### bing ### 
survey_senti <- survey_df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort=T) %>%
  ungroup()

#token contribution 
survey_bing<- survey_senti %>%
  group_by(sentiment) %>%
  top_n(5) %>%
  ungroup() %>%
  mutate(word=reorder(word, n)) %>%
  ggplot(aes(word, n, fill=sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y")+
  labs(y=" Token Contribution to Sentiment", x=NULL)+
  coord_flip()
ggplotly(survey_bing)

### nrc ### 
survey_senti_nrc <- survey_df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort=T) %>%
  ungroup()

#surey nrc wordcloud
survey_senti_nrc %>%
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort=TRUE) %>%
  acast(word ~sentiment, value.var="n", fill=0) %>%
  comparison.cloud(colors = c("grey20", "gray80"),
                   title.colors=c("red","blue"),
                   max.words=100, fixed.asp=TRUE,
                   scale=c(1.1,1.1), title.size=1.1, rot.per=0.5)


###########################################
######### ZIPF's law and tf idf ###########
###########################################

original_survey <- survey_df %>%
  unnest_tokens(word, text) %>%
  count(question, word, sort=TRUE) %>%
  ungroup()

total_words <- original_survey %>%
  group_by(question) %>%
  summarize(total=sum(n))

#joining survey words to each other
survey_words <- left_join(original_survey, total_words)

######################################
########## ZIPF's law ################
######################################

#creating a rank made of frequency
freq_by_rank <- survey_words %>%
  group_by(question) %>%
  mutate(rank = row_number(),
         `term frequency` = n/total)

#ZIPFs law plot with tangent line
freq_by_rank %>%
  ggplot(aes(rank, `term frequency`, color=question))+
  #let's add a tangent line , the first derivative, and see what the slop is
  geom_abline(intercept=-0.62, slope= -1.1, color='gray50', linetype=2)+
  geom_line(size= 1.1, alpha = 0.8, show.legend = FALSE)+
  scale_x_log10()+
  scale_y_log10()

###################################################
################# TF_IDF ##########################
###################################################
survey_words <- survey_words %>%
  bind_tf_idf(word, question, n)

survey_words %>%
  arrange(desc(tf_idf))%>%
  filter(n<2)

#graphing the token frequency
survey_tf <-survey_words %>%
  anti_join(stop_words)%>%
  arrange(desc(tf_idf)) %>%
  mutate(word=factor(word, levels =rev(unique(word)))) %>%
  group_by(question) %>%
  filter(n<5)%>%
  top_n(10) %>%
  ungroup() %>%
  ggplot(aes(word, tf_idf, fill=question))+
  geom_col(show.legend=FALSE)+
  labs(x=NULL, y="tf-idf")+
  facet_wrap(~question, ncol=2, scales="free")+
  coord_flip()
survey_tf

############ SHINY APP 
##### Cuisine word cloud
cuisine_wc <- survey_tidy %>%
  filter(question == "1")

# Removing custom stop words for word cloud
customstopwords <- tribble(~word, ~lexicon,
                           "prefer", "CUSTOM",
                           "food", "CUSTOM",
                           "cuisine", "CUSTOM",
                           "cuisines", "CUSTOM",
                           "love", "CUSTOM",
                           "lot", "CUSTOM",
                           "eating", "CUSTOM",
                           "cooking", "CUSTOM",
                           "um", "CUSTOM",
                           "weird", "CUSTOM",
                           "yeah", "CUSTOM",
                           "started", "CUSTOM",
                           "studying", "CUSTOM",
                           "economy", "CUSTOM",
                           "changed", "CUSTOM",
                           "avoid", "CUSTOM",
                           "preferences", "CUSTOM",
                           "logical", "CUSTOM")

cuisine_wc <- cuisine_wc %>%
  anti_join(customstopwords)

cuisine_freq <- cuisine_wc %>%
  count(word, sort=T)

wordcloud2(cuisine_freq, size = 1, minSize = 30, color = 'random-dark', backgroundColor = "white",
           minRotation = 0, maxRotation = 0)

##### Pairing word cloud
pairing_wc <- survey_tidy %>%
  filter(question == "3")

pairing_freq <- pairing_wc %>%
  count(word, sort=T)

wordcloud2(pairing_freq, size = 1, minSize = 100, color = 'random-dark', backgroundColor = "white",
           minRotation = 0, maxRotation = 0)

# DASHBOARD
ui <- dashboardPage(
  dashboardHeader(title = "Business Insights"),
  dashboardSidebar(
    sidebarMenu(
      id = "tabs",
      menuItem("Cuisine", tabName = "cuisine"),
      menuItem("Drinks and Occasions", tabName = "pairing"),
      menuItem("Correlation Network", tabName = "network")
    )
  ),
  dashboardBody(
    tabItems(
      # First tab 
      tabItem(tabName = "cuisine",
              fluidRow(
                box(width = 6,
                    title = "Cuisine Word Cloud",
                    sliderInput("cuisinefreq", "Minimum Frequency",
                                min = 10,  max = 100, value = 30)),
                box(width = 12,
                    wordcloud2Output("wc1")))),
      # Second tab 
      tabItem(tabName = "pairing",
              fluidRow(
                box(width = 6,
                    title = "Drinks and Occasions Word Cloud",
                    sliderInput("pairingfreq", "Minimum Frequency",
                                min = 10,  max = 100, value = 30)),
                box(width = 12,
                    wordcloud2Output("wc2")))),
      # Third tab 
      tabItem(tabName = "network",
              fluidRow(
                box(width = 6,
                    title = "Correlation Network",
                    sliderInput("corrcoef",
                                "Correlation Coefficient",
                                min = 0,  max = 1, value = 0.6)),
                box(width = 12,
                    plotOutput("corrnetwork"))))
    )
  )
)

server <- function(input, output){
  # Output 1
  output$wc1 <- renderWordcloud2({
    wordcloud2(cuisine_freq, size = 1, minSize = input$cuisinefreq, color = 'random-dark', backgroundColor = "white",
               minRotation = 0, maxRotation = 0)})
  
  # Output 2
  output$wc2 <- renderWordcloud2({
    wordcloud2(pairing_freq, size = 1, minSize = input$pairingfreq, color = 'random-dark', backgroundColor = "white",
               minRotation = 0, maxRotation = 0)})
  
  # Output 3
  output$corrnetwork <- renderPlot({
    survey_cors %>%
      group_by(item1) %>%
      filter(correlation > input$corrcoef) %>%
      graph_from_data_frame() %>%
      ggraph(layout = "fr")+
      geom_edge_link(aes(edge_alpha = correlation), show.legend=F)+ #shows links between
      geom_node_point(color = "purple", size=6)+ #similar to ngram semantic method/template
      geom_node_text(aes(label=name), repel=T)+
      theme_void()
  })
}

shinyApp(ui, server)

