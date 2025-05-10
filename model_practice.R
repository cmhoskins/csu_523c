library(tidyverse)
library(tidymodels)

covid_url <- "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"

pop_url <- "co-est2023-alldata.csv"

data = readr::read_csv(covid_url, show_col_types = FALSE)

census = readr::read_csv(pop_url, show_col_types = FALSE) |>
  filter(COUNTY == "000") |>
  mutate(fips = STATE) |>
  select(fips, contains("2021"))

state_data <- data |>
  group_by(fips) |>
  mutate(new_cases = pmax(0, cases - lag(cases)),
         new_deaths = pmax(0, deaths - lag(deaths))) |>
  ungroup() |>
  left_join(census, by = "fips") |>
  mutate(m = month(date), y = year(date),
         season = case_when(
           m %in% 3:5 ~ "Spring",
           m %in% 6:8 ~ "Summer",
           m %in% 9:11 ~ "Fall",
           m %in% c(12,1,2) ~ "Winter")) |>
  group_by(state, y, season) |>
  mutate(season_cases = sum(new_cases, na.rm = TRUE),
         season_death = sum(new_deaths, na.rm = TRUE)) |>
  distinct(state, y, season, .keep_all = TRUE) |>
  ungroup() |>
  drop_na() |>
  mutate(logC = log(season_cases +1))

skimr::skim(state_data)

## Data Splitting

set.seed(123)
split <- initial_split(state_data, prop = .8, strata = season)
s_train <- training(split)
s_testing <- tesing(split)
s_folds <- vfold_cv(s_train, v=10)

## Feature Engineering

rec <- recipe(logC ~ ., data = s_train) |>
  step_rm(season_cases, state) |>
  step_dummy(all_nominal_predictors()) |>
  step_scale(all_numberic_predictors()) |>
  step_center(all_numberic_predictors())

## Define Models
lm_mod <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

rf_mod <- rand_forest() |>
  set_engine("ranger", importance = 'impurity') |>
  set_mode("regression")

b_mod <- boost_tree() |>
  set_engine("xgboost") |>
  set_mode("regression")

nn_mod <- mlp() |>
  set_engine("nnet") |>
  set_mode("regression")

## Workflow Set
wf <- workflow_set(list(rec), list(lm_mod, rf_mod, b_mod, nn_mod))

workflow_map(wf, resamples = s_folds)

## Select
autoplot(wf) + theme_linedraw()

fit <- workflow() |>
  add_recipe(rec) |>
  add_model(rf_mod) |>
  fit(data = s_train)

## VIP

vip::vip(fit)

## Metrics / Predictions

predictions = augment(fit, new_data = s_test)

metrics(predictions, truth = logC, estimate = .pred)

ggplot(predictions, aes(x = logC, y = .pred)) +
  geom_point() +
  geom_abline() +
  geom_smooth(method)
