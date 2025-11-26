############################################################
# Load packages
############################################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)


############################################################
# Download and load MovieLens 10M
############################################################
dl<-"ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file<-"ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file<-"ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings<-as.data.frame(
  str_split(read_lines(ratings_file), fixed("::"), simplify=TRUE),
  stringsAsFactors=FALSE
)

colnames(ratings)<-c("userId", "movieId", "rating", "timestamp")
ratings<-ratings %>%
  mutate(
    userId=as.integer(userId),
    movieId=as.integer(movieId),
    rating=as.numeric(rating),
    timestamp=as.integer(timestamp)
  )

movies<-as.data.frame(
  str_split(read_lines(movies_file), fixed("::"), simplify=TRUE),
  stringsAsFactors=FALSE
)
colnames(movies)<-c("movieId", "title", "genres")
movies<-movies%>%mutate(movieId=as.integer(movieId))

movielens<-left_join(ratings, movies, by="movieId")


############################################################
# Create edx and final_holdout_test sets
############################################################
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)


############################################################
# Exploratory Data
############################################################
library(ggplot2)

cat("Number of ratings:", format(nrow(edx), big.mark=","), "\n")
cat("Number of users:", length(unique(edx$userId)), "\n")
cat("Number of movies:", length(unique(edx$movieId)), "\n")

# Summary of ratings
summary(edx$rating)

# Rating distribution
ggplot(edx, aes(rating))+
  geom_histogram(binwidth=0.5, color="black", fill="lightblue") +
  ggtitle("Distribution of Movie Ratings") +
  xlab("Rating")+ 
  ylab("Count")+
  theme_minimal()

# Number of ratings per user
user_count<-edx%>%
  group_by(userId)%>%
  summarize(n=n())

ggplot(user_count, aes(n))+
  geom_histogram(bins = 40, color="black", fill="orange")+
  scale_x_log10()+
  ggtitle("Number of Ratings per User")+
  xlab("Number of Ratings (log10)")+
  ylab("User Count")+
  theme_minimal()

# Number of ratings per movie
movie_count<-edx%>%
  group_by(movieId)%>%
  summarize(n = n())

ggplot(movie_count, aes(n))+
  geom_histogram(bins = 40, color="black", fill = "lightgreen")+
  scale_x_log10()+
  ggtitle("Number of Ratings per Movie (Log Scale)")+
  xlab("Number of Ratings (log10)")+
  ylab("Movie COunt")+
  theme_minimal()

# Average movie rating vs. number of ratings
movie_avgs<-edx%>%
  group_by(movieId)%>%
  summarize(avg_rating=mean(rating), n=n())

ggplot(movie_avgs, aes(n, avg_rating))+
  geom_point(alpha=0.3)+
  scale_x_log10()+
  geom_smooth()+
  ggtitle("Average Rating vs. Number of Ratings per Movie")+
  xlab("Number of Ratings (log10)")+
  ylab("Average Rating")+
  theme_minimal()

# Average movie rating per user
user_avgs<-edx%>%
  group_by(userId)%>%
  summarize(avg_rating=mean(rating), n=n())

ggplot(user_avgs, aes(avg_rating))+
  geom_histogram(binwidth = 0.1, color="black", fill="purple")+
  ggtitle("Distribution of Average Ratings per User")+
  xlab("Average Rating Given by User")+
  ylab("Count of Users")+
  theme_minimal()


############################################################
# Ratings Over Time
############################################################
library(lubridate)

edx<-edx %>%
  mutate(date=as_datetime(timestamp))

ratings_by_month<-edx %>%
  mutate(month=floor_date(date, "month"))%>%
  group_by(month)%>%
  summarize(n=n())

ggplot(ratings_by_month, aes(month, n))+
  geom_line(color="steelblue")+
  ggtitle("Number of Ratings Over Time")+
  xlab("Month") +
  ylab("Number of Ratings")+
  theme_minimal()


############################################################
# Regularized Movie + User Effects Model 
############################################################
edx<-edx%>%
  mutate(year=str_extract(title, "\\((\\d{4})\\)$")%>%
           str_remove_all("[()]")%>%as.numeric())

lambda<-5
mu<-mean(edx$rating)

# Movie effect
b_i<-edx%>%
  group_by(movieId)%>%
  summarize(b_i=sum(rating - mu)/(n()+lambda))

# User effect
b_u<-edx%>%
  left_join(b_i, by="movieId") %>%
  group_by(userId)%>%
  summarize(b_u=sum(rating-mu-b_i)/(n()+lambda))

# RMSE function
RMSE<-function(true, predicted){
  sqrt(mean((true-predicted)^2))
}


############################################################
# Predict on edx (training RMSE)
############################################################
pred_train<-edx%>%
  left_join(b_i, by="movieId")%>%
  left_join(b_u, by="userId")%>%
  mutate(pred=mu+b_i+b_u)

rmse_train<-RMSE(pred_train$rating, pred_train$pred)
cat("Training RMSE:", rmse_train, "\n")


############################################################
# Predict on final_holdout_test (final RMSE)
############################################################
pred_holdout<-final_holdout_test%>%
  left_join(b_i, by="movieId")%>%
  left_join(b_u, by="userId")%>%
  mutate(pred=mu+b_i+b_u)

# Clamp predictions
pred_holdout<-pred_holdout %>%
  mutate(pred=pmin(pmax(pred, 0.5), 5))

rmse_holdout< RMSE(pred_holdout$rating, pred_holdout$pred)
cat("Final Holdout RMSE:", rmse_holdout, "\n")


############################################################
# Top 5 Recommended Movies for a New User
############################################################
best_movies<-b_i%>%
  left_join(movies, by="movieId")%>%
  mutate(pred=mu+b_i)%>%
  arrange(desc(pred))%>%
  slice_head(n=5)

best_movies
