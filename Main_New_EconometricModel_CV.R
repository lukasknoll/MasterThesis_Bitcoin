library(httr)
library(jsonlite)
library(tidyverse)
library(ggplot2)
library(modeltime)
library(parsnip)
library(rsample)
library(recipes)
library(prophet)
library(glmnet)
library(dplyr)
library(xts)
library(corrplot)
library(plotly)
library(tidymodels)
library(lubridate)
library(caret)
library(forecast)
library(prophet)

#Setting API key and parameters
api_key <- "APIKEY"
base_url <- "https://api.binance.com"
endpoint <- "/api/v3/klines"
symbol <- "BTCUSDT"
interval <- "1d"

start_time <- as.numeric(as.POSIXct("2014-05-16")) * 1000
end_time <- as.numeric(as.POSIXct("2024-05-17")) * 1000

btc_data <- data.frame()

interval_duration <- 500 * 24 * 60 * 60 * 1000

#Retrieving data from the API
while (start_time < end_time) {
  current_end_time <- min(start_time + interval_duration - 1, end_time)
  url <- paste0(base_url, endpoint, "?symbol=", symbol, "&interval=", interval, "&startTime=", start_time, "&endTime=", current_end_time)
  response <- GET(url, add_headers("X-MBX-APIKEY" = api_key))
  
  if (http_error(response)) {
    stop("HTTP request failed: ", http_status(response)$reason)
  }
  
  btc_json <- content(response, "text", encoding = "UTF-8")
  btc_data_temp <- fromJSON(btc_json)
  btc_data_temp <- as.data.frame(btc_data_temp)
  btc_data <- rbind(btc_data, btc_data_temp)
  
  start_time <- current_end_time + 1
}

#Specifying column names and convert to numeric
colnames(btc_data) <- c("timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore")
btc_data[] <- lapply(btc_data, as.numeric)

btc_data$timestamp <- as.POSIXct(btc_data$timestamp / 1000, origin = "1970-01-01")
btc_data$close_time<- as.POSIXct(btc_data$close_time / 1000, origin = "1970-01-01")

#EDA and correlation analysis
btc_data <- btc_data %>% select(-ignore)
corr_matrix <- cor(btc_data %>% select(-timestamp, -close_time))
corrplot(corr_matrix, method = "circle", tl.cex = 0.8, mar = c(0, 0, 1, 0))

#Splitting data into training and test sets
split_index <- floor(nrow(btc_data) * 0.8)
train_data <- btc_data[1:split_index, ]
test_data <- btc_data[(split_index + 1):nrow(btc_data), ]

splits <- time_series_split(btc_data, assess = "1 year", cumulative = TRUE)

#Plotting the time series split
splits %>%
  tk_time_series_cv_plan() %>%
  group_by(.id) %>%
  plot_time_series_cv_plan(timestamp, close,
                           .interactive = TRUE,
                           .title = "Bitcoin Closing Prices from 2017 to 2024")

average_close_price_test <- mean(test_data$close, na.rm = TRUE)

print(paste("The average closing price of the test data set is:", average_close_price_test))

set.seed(123)
cv_splits <- vfold_cv(train_data, v = 10)

#Defining recipes for each model
recipe_lm <- recipe(close ~ open + high + low + quote_asset_volume, data = train_data)
recipe_xgb <- recipe(close ~ open + high + low + quote_asset_volume, data = train_data)
recipe_svm <- recipe(close ~ open + high + low + quote_asset_volume, data = train_data)
recipe_arima <- recipe(close ~ timestamp + open + high + low + quote_asset_volume, data = train_data)
recipe_prophet <- recipe(close ~ timestamp + open + high + low + quote_asset_volume, data = train_data)

#Defining models
lm_model <- linear_reg() %>%
  set_engine("lm")

xgboost_model <- boost_tree(mode = "regression") %>%
  set_engine("xgboost")

svm_model <- svm_poly(mode = "regression") %>%
  set_engine("kernlab")

arima_model <- arima_reg() %>%
  set_engine("auto_arima")

prophet_model <- prophet_reg() %>%
  set_engine("prophet")

#Defining workflows
workflow_lm <- workflow() %>%
  add_recipe(recipe_lm) %>%
  add_model(lm_model)

workflow_xgb <- workflow() %>%
  add_recipe(recipe_xgb) %>%
  add_model(xgboost_model)

workflow_svm <- workflow() %>%
  add_recipe(recipe_svm) %>%
  add_model(svm_model)

workflow_arima <- workflow() %>%
  add_recipe(recipe_arima) %>%
  add_model(arima_model)

workflow_prophet <- workflow() %>%
  add_recipe(recipe_prophet) %>%
  add_model(prophet_model)

#Fitting with cv
fit_resamples_lm <- workflow_lm %>%
  fit_resamples(resamples = cv_splits, metrics = metric_set(rmse, rsq))

fit_resamples_xgb <- workflow_xgb %>%
  fit_resamples(resamples = cv_splits, metrics = metric_set(rmse, rsq))

fit_resamples_svm <- workflow_svm %>%
  fit_resamples(resamples = cv_splits, metrics = metric_set(rmse, rsq))

fit_resamples_arima <- workflow_arima %>%
  fit_resamples(resamples = cv_splits, metrics = metric_set(rmse, rsq))

fit_resamples_prophet <- workflow_prophet %>%
  fit_resamples(resamples = cv_splits, metrics = metric_set(rmse, rsq))

#Collecting metrics
metrics_lm <- fit_resamples_lm %>% collect_metrics()
metrics_xgb <- fit_resamples_xgb %>% collect_metrics()
metrics_svm <- fit_resamples_svm %>% collect_metrics()
metrics_arima <- fit_resamples_arima %>% collect_metrics()
metrics_prophet <- fit_resamples_prophet %>% collect_metrics()

#Combining metrics for comparison
all_metrics <- bind_rows(
  metrics_lm %>% mutate(model = "Linear Regression"),
  metrics_xgb %>% mutate(model = "XGBoost"),
  metrics_svm %>% mutate(model = "SVM"),
  metrics_arima %>% mutate(model = "ARIMA"),
  metrics_prophet %>% mutate(model = "Prophet")
)

ggplot(all_metrics %>% filter(.metric == "rmse"), aes(x = model, y = mean)) +
  geom_col() +
  labs(title = "Model Comparison - RMSE", y = "RMSE", x = "Model") +
  theme_minimal()

ggplot(all_metrics %>% filter(.metric == "rsq"), aes(x = model, y = mean)) +
  geom_col() +
  labs(title = "Model Comparison - R-Squared", y = "R-Squared", x = "Model") +
  theme_minimal()

#Final model training on the entire dataset
final_lm <- workflow_lm %>% fit(train_data)
final_xgb <- workflow_xgb %>% fit(train_data)
final_svm <- workflow_svm %>% fit(train_data)
final_arima <- workflow_arima %>% fit(train_data)
final_prophet <- workflow_prophet %>% fit(train_data)

#Modeltime table
model_tbl <- modeltime_table(
  final_arima,
  final_lm,
  final_prophet,
  final_xgb,
  final_svm
)

calibration_tbl <- model_tbl %>%
  modeltime_calibrate(test_data)

#Calculating and display model accuracies
accuracy_tbl <- calibration_tbl %>%
  modeltime_accuracy()

accuracy_tbl %>%
  table_modeltime_accuracy(resizable = TRUE, bordered = TRUE)

#Predictions
results <- test_data %>%
  select(timestamp, close) %>%
  rename(actual = close) %>%
  mutate(
    `Linear Regression` = predict(final_lm, test_data)$.pred,
    ARIMA = predict(final_arima, test_data)$.pred,
    Prophet = predict(final_prophet, test_data)$.pred,
    XGBoost = predict(final_xgb, test_data)$.pred,
    SVM = predict(final_svm, test_data)$.pred
  )

#Plotting
comparison_plot <- ggplot() +
  geom_line(data = btc_data, aes(x = timestamp, y = close, color = "Actual"), linewidth = 0.8) +
  geom_line(data = results, aes(x = timestamp, y = `Linear Regression`, color = "Linear Regression"), linewidth = 0.8) +
  geom_line(data = results, aes(x = timestamp, y = ARIMA, color = "ARIMA"), linewidth = 0.8) +
  geom_line(data = results, aes(x = timestamp, y = Prophet, color = "Prophet"), linewidth = 0.8) +
  geom_line(data = results, aes(x = timestamp, y = XGBoost, color = "XGBoost"), linewidth = 0.8) +
  geom_line(data = results, aes(x = timestamp, y = SVM, color = "SVM"), linewidth = 0.8) +
  labs(title = "Bitcoin Closing Prices: Actual vs Predicted",
       x = "Date", y = "Closing Price") +
  scale_color_manual(values = c("Actual" = "black",
                                "Linear Regression" = "blue",
                                "ARIMA" = "green",
                                "Prophet" = "red",
                                "XGBoost" = "orange",
                                "SVM" = "purple")) +
  theme_minimal()

interactive_plot <- ggplotly(comparison_plot) %>%
  layout(
    legend = list(title = list(text = "Model"),
                  itemclick = "toggle",
                  itemdoubleclick = "toggleothers")
  )

# Display interactive plot
interactive_plot

#________________________Creating forecasts________________________________________________

#Prophet modeling and prediction function
forecast_prophet <- function(df, periods = 365) {
  m <- prophet(df)
  future <- make_future_dataframe(m, periods = periods)
  forecast <- predict(m, future)
  yhat <- forecast$yhat[(nrow(forecast) - periods + 1):nrow(forecast)]
  yhat[yhat < 0] <- 0
  return(yhat)
}

future_periods <- 365

#Creating future values for each independent variable except close
future_data <- data.frame(
  timestamp = seq(max(btc_data$timestamp) + 1, by = "day", length.out = future_periods)
)

#Applying the Prophet model to each independent variable
future_data$open <- forecast_prophet(data.frame(ds = btc_data$timestamp, y = btc_data$open), future_periods)
future_data$high <- forecast_prophet(data.frame(ds = btc_data$timestamp, y = btc_data$high), future_periods)
future_data$low <- forecast_prophet(data.frame(ds = btc_data$timestamp, y = btc_data$low), future_periods)
future_data$volume <- forecast_prophet(data.frame(ds = btc_data$timestamp, y = btc_data$volume), future_periods)
future_data$quote_asset_volume <- forecast_prophet(data.frame(ds = btc_data$timestamp, y = btc_data$quote_asset_volume), future_periods)
future_data$taker_buy_base_asset_volume <- forecast_prophet(data.frame(ds = btc_data$timestamp, y = btc_data$taker_buy_base_asset_volume), future_periods)
future_data$taker_buy_quote_asset_volume <- forecast_prophet(data.frame(ds = btc_data$timestamp, y = btc_data$taker_buy_quote_asset_volume), future_periods)

print(head(future_data))

#Combining historical and future data
historical_data <- btc_data %>% select(timestamp, open, high, low, volume, quote_asset_volume, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, close)

combined_data <- bind_rows(
  historical_data,
  future_data %>% mutate(close = NA)
)

print(head(combined_data))
print(tail(combined_data))

#Readjustment of the models (combined_data)
lm_model_refit <- linear_reg() %>%
  set_engine("lm") %>%
  fit(close ~ open + timestamp + high + low + quote_asset_volume, data = combined_data)

arima_model_refit <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(close ~ open + timestamp + high + low + quote_asset_volume, data = combined_data)

prophet_model_refit <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(close ~ open + timestamp + high + low + quote_asset_volume, data = combined_data)

XgBoost_model_refit <- boost_tree(mode = "regression") %>%
  set_engine("xgboost") %>%
  fit(close ~ open + timestamp + high + low + quote_asset_volume, data = combined_data)

SVM_poly_model_refit <- svm_poly(mode = "regression") %>%
  set_engine("kernlab") %>%
  fit(close ~ open + timestamp + high + low + quote_asset_volume + taker_buy_quote_asset_volume, data = combined_data)

#Modeltime table creation
model_tbl_refit <- modeltime_table(
  arima_model_refit,
  lm_model_refit,
  prophet_model_refit,
  XgBoost_model_refit,
  SVM_poly_model_refit
)

#Calibration
calibration_tbl_refit <- model_tbl_refit %>%
  modeltime_calibrate(combined_data)

#Creating forecasts and plots
forecast_tbl_refit <- model_tbl_refit %>%
  modeltime_forecast(
    new_data = future_data,
    actual_data = combined_data,
    conf_interval = 0.80
  ) 

#Plotting the forecasts
forecast_tbl_refit %>%
  plot_modeltime_forecast(.interactive = TRUE,
                          .title = "Bitcoin Closing Prices: Forecast")

#_______________________________Volitality________________________________
  
#Calculation of historical annual volatility
historical_returns <- btc_data %>%
  mutate(log_return = log(close / lag(close))) %>%
  drop_na() %>%
  mutate(year = year(timestamp)) %>%
  group_by(year) %>%
  summarise(volatility = sd(log_return) * sqrt(252)) %>%
  ungroup()

#Calculation of the average annual volatility
average_historical_volatility <- mean(historical_returns$volatility, na.rm = TRUE)

#Calculation of the annual volatility of the forecasts
forecast_returns <- forecast_tbl_refit %>%
  filter(.key == "prediction") %>%
  mutate(log_return = log(.value / lag(.value))) %>%
  drop_na() %>%
  mutate(year = year(.index)) %>%
  group_by(year) %>%
  summarise(volatility = sd(log_return) * sqrt(252)) %>%
  ungroup()

#Calculation of average annual volatility of forecasts
average_forecast_volatility <- mean(forecast_returns$volatility, na.rm = TRUE)

#Showing results
cat("Average annual volatility of historical BTC data (time period = 7 years):", average_historical_volatility, "\n")
cat("Average annual volatility of the predicted BTC data (time period = 1 year):", average_forecast_volatility, "\n")

