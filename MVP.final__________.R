################################################################################
################################################################################

#  АНАЛИЗ ДЕГУСТАЦИОННОГО ДАТАСЕТА

################################################################################
################################################################################

# Устанавливаем необходимые пакеты и загружаем их
install.packages("psych")
install.packages("skimr", dependencies = TRUE)
library(SmartEDA)
install.packages(c("tidytext", "dplyr", "tidyr", "textdata"))
library(skimr)
install.packages("modeest")
install.packages("ranger")
install.packages("randomForest")
install.packages("caret")
install.packages(c("ranger", "randomForest"))
install.packages("pmml", "xml")
install.packages('xgboost')

install.packages("future.apply")
library(future.apply)
plan(multisession) # Использовать несколько ядер

# Установка необходимых пакетов (если они еще не установлены)
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("tidymodels")) install.packages("tidymodels")
if (!require("textrecipes")) install.packages("textrecipes")
if (!require("stopwords")) install.packages("stopwords")
if (!require("SnowballC")) install.packages("SnowballC")
if (!require("glmnet")) install.packages("glmnet")

# Загрузка библиотек
library(parallel)
library(stopwords)
library(SnowballC)
library(ranger)
library(randomForest)
library(vip)
library(modeest)
library(purrr)
library(stringr)
library(ggplot2)
library(tibble)
library(dplyr)
library(patchwork)
library(RColorBrewer)
library(ggrepel) 
library(tidyverse)
library(tidytext)
library(tidymodels)
library(textrecipes)
library(tidyr)
library(stats)
library(caret)
library(xgboost)
library(yardstick)

# Объединяем файлы по странам в единый датасет, и удаляем дубликаты записей,
# присваиваем индексам датафрейма значения идентификаторов записи, выясняем 
# сколько у нас стран.

setwd(".\datasets")
files <- list.files(pattern = "*.csv")
print(paste("Количество винодельческих стран:", as.integer(length(files))))
df <- files %>%
  map_dfr(~read.csv(.x, sep = ";")) 

###############################################################################
###############################################################################

# ВЫПОЛНЕНИЕ ЗАДАНИЯ, ЧАСТЬ № 1

###############################################################################
###############################################################################

# Задание № 1.1: 
# 1.1.1 Какое количество строк в загруженном dataframe?
# 1.1.2. Какое количество столбцов в загруженном dataframe?
cat(" Количество записей:", nrow(df), "\n", "Количество признаков:", ncol(df))

################################################################################
# Задание № 1.2 Очистка данных:

# 1.2.1. Какое количество строк в очищенном dataframe?
# 
# Для начала проверим предположение о том, что у нас есть дублирующиеся записи,
duplicated <- df %>%
  group_by(entry_id)%>%
  summarize(n=n())%>%
  filter(n>1)
print(paste("Процент дублирующихся записей:", 
            round(100*length(duplicated)/length(df), 1)))

# Избавляемся от дубликатов и смотрим количество записей. 
df <- df[!duplicated(df$entry_id),]
cat(" Количество записей:", nrow(df))

################################################################################
# 1.2.2. Какое количество уникальных значений содержится в taster_name (вкл. NA )
  
rpl=c("а" = "a", "о" = "o", "е" = "e")
# rpl=c("а" = "a", "б" = "b", "с" = "c", "е" = "e", "ё" = "e",
#   "к" = "k", "м" = "m", "н" = "h", "о" = "o", "р" = "p",
#   "т" = "t", "у" = "y", "х" = "x", "А" = "A", "В" = "B",
#   "С" = "C", "Е" = "E", "Ё" = "E", "К" = "K", "М" = "M",
#   "Н" = "H", "О" = "O", "Р" = "P", "Т" = "T", "У" = "Y",
#   "Х" = "X")

# осуществляем замену символов 
df <- df %>%
  mutate(across(where(is.character),
                ~ str_replace_all(., rpl)))

cat("Количество дегустаторов включая анонимных:", n_distinct(df$taster_name))

###############################################################################
# 1.3 ЭКСПЛОРАТОРНЫЙ АНАЛИЗ
###############################################################################

# 1.3.1. Для переменных points и price приведите набор описательных статистик:
#   * Среднее
#   * Медиана
#   * Стандартное отклонение

avmedsd <- function(data, feature) {
  data %>%
    summarize(
      average = mean({{feature}}, na.rm = TRUE),
      median = median({{feature}}, na.rm = TRUE),
      stdev = sd({{feature}}, na.rm = TRUE)
    ) %>% 
    round(1)
}

avmedsd(df, price)
avmedsd(df, points)

# Как мы видим, стандартное отклонение больше среднего и медианного значения.
# Поскольку медиана значительно меньше среднего значения, мы можем уверенно
# предполагать о наличии у распределения тяжёлых хвостов справа. 
# 
# Строим гистограмму
hist(df$price, breaks=100)

quantile(df$price, 0.995, 'na.rm' = TRUE)
quantile(df$price, 0.005, 'na.rm' = TRUE)
quantile(df$price, 0.975, 'na.rm' = TRUE)
quantile(df$price, 0.025, 'na.rm' = TRUE)

# 99% наблюдений находится в диапазоне 7...230 $,
# при этом 95% лежит в диапазоне 10...110 $
# Строим гистограмму с логарифмической шкалой

ggplot(df, aes(x = price)) +
  geom_histogram(fill = "skyblue", bins = 30) +
  scale_x_log10() +  # Логарифмическая шкала по оси X
  labs(title = "Гистограмма цен (логарифмическая шкала)",
       x = "Цена, log10(USD)", 
       y = "Количество вин") +
  theme_minimal()

# Несмотря на близость к логнормальному распределению, наше распределение цены 
# всё же имеет тяжёлых хвост справа.

################################################################################

# ЗАДАНИЕ 1.3.2 Опишите связь между points и price
# * Укажите коэффициент корреляции и p value

# Поскольку у нас тяжёлые хвосты, использована будет ранговая корреляция Спирмена,
# и мы не будем осуществлять преобразования переменной price

res <- cor.test(x, y, method = "spearman")
res$p.value  # Может быть 0 из-за округления
format(res$p.value, scientific = TRUE, digits = 20)

# Проверка корреляции с выводом пlog()# Проверка корреляции с выводом полной статистики
corr_test <- cor.test(
  x = df$points,
  y = df$price,
  method = "spearman",
  use = "complete.obs",  # Игнорировать NA
  exact = FALSE  # Для больших выборок (>500)
)
corr_test$p.value
cat(
  "Корреляция Спирмена между рейтингом и ценой:\n",
  "Коэффициент (ρ): ", round(corr_test$estimate, 3), "\n",
  "p-value: ", ifelse(corr_test$p.value < 0.001, "<0.001", round(corr_test$p.value, 3)), "\n",
  "N наблюдений: ", sum(complete.cases(df[, c("points", "price")]))
)

################################################################################
# 1.4 Визуализация
################################################################################

# Для отображения характера распределения значений оценок и цен, построим
# гистограммы, поскольку они у нас нет задачи сопоставления распределений, 
# а наличие очень тяжёлых хвостов у цены вина приводит к тому, что ящик с усами
# схлопывается в линию и теряет всякую информативность. А для оценок вина в 
# баллах,форма распределения, нам интереснее чем визуализация медианы и границы 
# распределения.

# Распределение 'points' - Гистограмма
ggplot(df, aes(x = points)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Распределение оценок вин", x = "Оценки", y = "Частота")
ggsave("points_distribution.png")

# Распределение 'price' - Гистограмма с логарифмической шкалой
ggplot(df, aes(x = price)) +
  geom_histogram(bins = 30, fill = "salmon", color = "black") +
  scale_x_log10() +
  labs(title = "Распределение цен на вина", x = "Цена (USD)", y = "Частота")
ggsave("price_distribution.png")

# Связь между 'points' и 'price' - Точечная диаграмма
ggplot(df, aes(x = price, y = points)) +
  geom_point(alpha = 0.5, color = "purple") +
  labs(title = "Связь между оценками и ценой", x = "Цена (USD)", y = "Оценки")
ggsave("points_price_relationship.png")

# Ожидаемо, мы получаем большое количество точек и видим что с ростом оценок, 
# цены тоже растут. Рассматривать зависимость в таком виде неудобно, поэтому мы 
# аггрегируем медианным значением цены по каждой оценке.

median_prices <- df %>%
  group_by(points) %>%
  summarise(median_price = median(price, na.rm = TRUE))

# Строим график
# ggplot(median_prices, aes(x = median_price, y = points)) +
ggplot(median_prices, aes(x = median_price, y = points)) +
  geom_point(size = 3, color = "steelblue") +
  geom_line(color = "grey80", alpha = 0.7) +  # Добавляем линию для наглядности
  labs(
    title = "Зависимость медианной цены от оценки вина",
    subtitle = "Каждая точка представляет медианную цену для соответствующей оценки",
    x = "Оценка (points)",
    y = "Медианная цена (USD)"
  ) +
  theme_minimal() +
  scale_y_continuous(labels = scales::dollar_format())  # Форматируем ось Y в денежном формате


# Мы видим что зависимость оценки вина от медианного значения его цены, хорошо
# укладывается в зависимость с резким ростом и постепенным его замедлением и
# выходом на плато, что близко к логарифмической функции.

################################################################################
################################################################################

# ЧАСТЬ № 2. ТВОРЧЕСКАЯ

################################################################################
################################################################################

# ЗАДАЧИ ЗАКАЗЧИКА:

################################################################################
# * Имена ТОП-5 самых продуктивных авторов отзывов (по числу отзывов):

# Получаем топ-5 ревьюеров и их топ-5 стран
top_data <- df %>%
  count(taster_name, sort = TRUE, name = "total_reviews") %>%
  filter(!is.na(taster_name)) %>%
  slice_max(total_reviews, n = 5) %>%
  mutate(color_data = map(taster_name, ~ {
    df %>%
      filter(taster_name == .x) %>%
      count(country, sort = TRUE, name = "reviews") %>%
      mutate(
        rank = row_number(),
        country = if_else(rank <= 5, country, "Others"),
        country = fct_reorder(country, -reviews)
      ) %>%
      group_by(country) %>%
      summarise(reviews = sum(reviews), .groups = "drop") %>%
      mutate(
        pct = round(reviews/sum(reviews)*100, 1),
        label = if_else(country != "Others", paste0(country, "\n", pct, "%"), NA_character_)
      )
  })) %>%
  select(-total_reviews) %>%
  unnest(color_data)

# Создаем палитру
pastel_palette <- c(
  "#FFD1DC",  # Розовый
  "#B5EAD7",  # Мятный
  "#C7CEEA",  # Лавандовый
  "#E2F0CB",  # Салатовый
  "#FFDAC1",  # Персиковый
  "#B2B2B2"   # Серый для Others
)

# Дополнительные цвета, если стран больше 5
extra_pastels <- c(
  "#F8B195", "#F67280", "#C06C84", "#6C5B7B", "#355C7D",
  "#99B898", "#FECEAB", "#FF847C", "#E84A5F", "#2A363B"
)

# Собираем полную палитру
all_countries <- top_data %>%
  filter(country != "Others") %>%
  pull(country) %>%
  unique() %>%
  sort()

country_colors <- setNames(
  c(pastel_palette[1:min(5, length(all_countries))],
    extra_pastels[1:max(0, length(all_countries) - 5)]),
  all_countries
)
country_colors["Others"] <- "#D3D3D3"  # Светло-серый для Others

# Функция построения диаграммы
create_pie <- function(reviewer_name) {
  plot_data <- top_data %>% filter(taster_name == reviewer_name)
  
  ggplot(plot_data, aes(x = "", y = reviews, fill = country)) +
    geom_col(width = 1, color = "white", alpha = 0.8, show.legend = FALSE) +
    coord_polar("y", start = 0) +
    geom_label_repel(
      aes(label = label),
      # Убрано position = position_identity()
      nudge_x = 1.5,  # Смещение по горизонтали от центра
      nudge_y = 0,    # Смещение по вертикали
      size = 3,
      color = "gray30",
      fill = alpha("white", 0.7),
      show.legend = FALSE,
      box.padding = 0.5,  # Отступ во избежание наложения
      point.padding = 0.3, # Отступ от точки привязки
      segment.color = "gray50", # Цвет линий
      segment.size = 0.2,       # Толщина линий
      direction = "both",       # Движение меток в обе стороны
      max.overlaps = Inf
    ) +
    scale_fill_manual(values = country_colors) +
    labs(
      title = paste0(reviewer_name, "\n(n=", sum(plot_data$reviews), ")"),
      fill = NULL
    ) +
    theme_void() +
    theme(
      plot.title = element_text(size = 10, hjust = 0.5, face = "bold", 
                                color = "gray30"),
      panel.background = element_rect(fill = "transparent", color = NA),
      plot.background = element_rect(fill = "transparent", color = NA)
    )
}

# Создание и компоновка графиков
plots <- top_data %>%
  distinct(taster_name) %>%
  pull(taster_name) %>%
  map(create_pie)

final_plot <- wrap_plots(plots, ncol = 3) +
  plot_annotation(
    title = "Топ-5 стран в отзывах для каждого ревьюера",
    subtitle = "Остальные страны объединены в 'Others'",
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14, 
                                color = "gray30"),
      plot.subtitle = element_text(hjust = 0.5, size = 12, color = "gray50"),
      plot.background = element_rect(fill = "#F9F9F9", color = NA)
    )
  ) +
  plot_layout(guides = "collect")

# 5. Выводим и сохраняем результат
print(final_plot)
ggsave("reviewers_top5_pastel.png", final_plot, 
       width = 12, height = 8, dpi = 300, bg = "transparent")

################################################################################
  
# Отличаются ли ценовые диапазон вин, которые оценивает каждый из ТОП-5?

top_reviewers <- df %>%
  count(taster_name, sort = TRUE) %>%
  filter(!is.na(taster_name)) %>%
  head(5) %>%                      #
  pull(taster_name)

filtered_df <- df %>% filter(taster_name %in% top_reviewers)

# A. Боксплот с логарифмической осью Y
plot_a <- ggplot(filtered_df, aes(x = taster_name, y = price)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(title = "A: Логарифмическая шкала", x = NULL, y = "Price (log10)")

# E. Разделенные графики: основной диапазон + выбросы
plot_e_main <- ggplot(filtered_df, aes(x = taster_name, y = price)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0, quantile(filtered_df$price, 0.99, 'na.rm' = TRUE))) +
  labs(title = "E1: Основной диапазон", x = NULL, y = "Price")

plot_e_outliers <- ggplot(filtered_df %>% filter(price > 
                                                   quantile(filtered_df$price, 
                                                            0.99, 'na.rm' = TRUE)), 
                          aes(x = taster_name, y = price)) +
  geom_jitter(width = 0.2) +
  labs(title = "E2: Выбросы (топ 10%)", x = NULL, y = "Price")

(plot_a)/ (plot_e_main | plot_e_outliers)


###############################################################################
###############################################################################

# ДОПОЛНИТЕЛЬНЫЕ ИССЛЕДОВАНИЯ

###############################################################################
###############################################################################


###############################################################################
# ПОДГОТОВКА ПРИЗНАКОВ
###############################################################################

# Проводим дополнительные исследования для отбора перспективных признаков для
# Построения модели, позволяющей прогнозировать оценку вина исходя из значения
# этих признаков. Также продолжаем исследовать данные и выявлять закономерности.

# Текст отзыва о вине как правило очень длинный, создадим служебный образец 
# (начальный фрагмент описания) чтобы было легче ориентироваться 

df$description_short <- substr(df$description, 1, 30)
df$description_short

# А теперь посмотрим, имеются ли у нас:
# 1) Оценки одного и того же вина разными дегустаторами
# 2) Повторные дегустации дегустатором конкретного вина
tastings <- df %>% 
  group_by(title) %>% 
  summarize(
    tasters = paste(taster_name, collapse="; "),
    unique_tasters = paste(unique(taster_name), collapse="; "),
    count = n(), un_taster_num = n_distinct(taster_name)
  ) %>% 
  filter(count > 1) %>% 
  arrange(desc(count))  # сортировка по убыванию количества
print(head(tastings, 20))

# Как мы видим, некоторые вина оценивались несколькими дегустаторами, причём
# иногда имели место неоднократные повторные дегустации. Также мы видим что 
# часть дегустаций была проведена без идентификации дегустатора. Посмотрим, 
# отличались ли описания вкусовых ощущений? Возьмём для примера вино 
# "Segura Viudas NV Aria # Estate Extra Dry Sparkling (Cava)" Которое 
# дегустировал только один дегустатор Michael Schachner

wine_name="Segura Viudas NV Aria Estate Extra Dry Sparkling (Cava)"
Segura_Viudas<- df%>% 
  filter(title==wine_name) %>%
  # select(description, title, taster_name)
  select(entry_id, description, title, points)%>%
  arrange(entry_id)
print(Segura_Viudas)

# Видно что оценки вкусовых качеств одного и того же вина (или вкусовое 
# восприятие), сильно варьируются не только от дегустатора к дегустатору, но и
# от дегустации к дегустации экспертом вина одного и того же наименования. 

# Определяем количество уникальных значений/ классов
uniques=map_int(df, ~length(unique(.x)))
print(uniques)
cat("Количество дегустаторов:",uniques[["taster_name"]])

# Рассматриваем тип имеющихся переменных
features<-data.frame(Type = sapply(df, class))
features

# Мы видим что в названии вина встречается год его урожая, выделим этот признак
# Создаём функцию извлечения корректных годов
extract_valid_years <- function(text) {
  # Ищем все 4-значные числа в тексте
  # all_numbers <- str_extract_all(text, "\\b\\d{4}\\b")[[1]] %>% as.numeric()
  all_numbers <- unlist(str_extract_all(text, "\\b\\d{4}\\b")) %>%
    max() %>% as.numeric()
    
  valid_years <- all_numbers[all_numbers >= 1850 & all_numbers <= 2050]
  # valid_years<-all_numbers
  # Возвращаем первый корректный год или NA если нет подходящих
  if(length(valid_years) > 0) {
    return(valid_years[1])  # берем первый год, если их несколько
  } else {
    return(NA)
  }
}

# Применяем функцию к записям в названии отзыва содержащего год урожая, и к
# записям самого отзыва, содержащим срок годности (ограничивающий год).
df$years <- sapply(df$title, extract_valid_years, USE.NAMES = FALSE)
df$bbyears <- sapply(df$description, extract_valid_years, USE.NAMES = FALSE)
# df$title[is.na(df$years)]

 # Посмотрим сколько у нас вин с годом урожая 30 и более лет назад
df %>%
  filter(years<1995)%>%
  select(title, winery, years, price)%>%
  arrange(years)

# Мы видим что вина урожая до 1934 года, стоят очень дёшево, что невозможно 
# Смотрим сколько у нас таких вин 
df %>%
  filter(years<1934) %>%
  summarize(n=n())

# Вин немного, будем считать, что по этим винам у нас нет цен 
df$price[df$years<1934] <-NA

# Смотрим продолжительность жизни вина (разность между годом до которого вино
# рекомендуется к потреблению и года его изготовления), проверяем на наличие 
# отрицательных значений, и если такие будут найдены, 
df$wine_life <- df$bbyears-df$years
df %>%
  filter(wine_life<0) %>%
  summarize(n=n())
df %>%
  filter(wine_life<0) %>%
  select(description, years, bbyears) %>%
  head(5)

# Как мы видим, все эти случаи относятся либо к упоминанию возраста виноградника,
# либо сопоставление опыта более ранней и текущей дегустации. Обнулением записи 
# для которых наблюдаем отрицательные продолжительности жизни вина.
df$bbyears[df$wine_life<0] <- NA
df$wine_life[df$wine_life<0] <- NA

# Посмотрим как распределены рекомендуемые сроки потребления вин
hist(df$wine_life, breaks=30)
# Распределение близко к распределению релея, максимальная плотность наблюдается 
# для срока жизни 6 лет, вин старше 30 лет очень мало

# Посмотрим как работают дегустаторы и вина каких ценовых диапазонов они 
# дегустируют
df %>%
  filter(!is.na(points)) %>%
  ggplot(aes(x = factor(taster_name), y = points, fill = taster_name)) +
  geom_violin(show.legend = TRUE) +
  # geom_jitter(width = 0.2, alpha = 0.01) +  # Добавляем точки данных
  # labs(title = "Распределение оценок по дегустаторам",
  #      x = "Дегустатор",
  #      y = "Оценка (points)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# Как мы видим поведение у дегустаторов разнится. 

df %>%
  filter(!is.na(price)) %>%
  ggplot(aes(x = factor(taster_name), y = price, fill = taster_name)) +
  geom_violin(show.legend = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.01) +  # Добавляем точки данных
  labs(title = "Распределение оценок по дегустаторам",
       x = "Дегустатор",
       y = "Оценка (points)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

summary(df$wine_life)
hist(df$wine_life, breaks = 10)

# Как мы видим чаще всего всречаются вина со сроком жизни 6 лет, а вина старше 
# 30 лет представлены отдельными дорогими винами, самое старое из которых имеет 
# возраст 80 лет.


library(ggplot2)
library(patchwork)

# Распределение оценок по дегустаторам
p1 <- df %>%
  filter(!is.na(points)) %>%
  ggplot(aes(x = factor(taster_name), y = points, fill = taster_name)) +
  geom_violin(show.legend = TRUE) +
  labs(title = "Распределение цен по дегустаторам") +
  theme(
    axis.text.x = element_blank(),  
    axis.title.x = element_blank(), 
    axis.ticks.x = element_blank()) 

# Распределение цен по дегустаторам
p2 <- df %>%
  filter(!is.na(price)) %>%
  ggplot(aes(x = factor(taster_name), y = price, fill = taster_name)) +
  geom_violin() +
  geom_jitter(width = 0.2, alpha = 0.01) +
  labs(title = "Распределение цен по дегустаторам",
       x = "Дегустатор",
       y = "Цена (price)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Компоновка графиков
combined_plot <- p1 / p2

combined_plot

# Мы видим,что низкие оценки у некоторых дегустаторов сочетаются с низкими 
# максимальными ценам на вино которое они пробуют, возможно большего это вино
# и не заслуживает?

# Посмотрим какое влияние оказывает продолжительность жизни вина, на оценки его 
# экспертами:
df %>%
  group_by(AgeGroup = cut(wine_life, 
                          breaks = c(NA, 0, 3, 6, 9, 12, 18, 24, 30, 80),
                          include.lowest = TRUE)) %>%
  summarize(n = n(),
            min_points = min(points, na.rm = TRUE),
            p25_points = quantile(points, probs = 0.25, na.rm = TRUE),
            median_points = median(points, na.rm = TRUE),
            p75_points = quantile(points, probs = 0.75, na.rm = TRUE),
            median_price = median(price, na.rm = TRUE),
            p25_price=quantile(price,probs = 0.25,na.rm = TRUE),
            p75_price = quantile(price, probs = 0.75, na.rm = TRUE)) %>%
  arrange(median_points)

# Ожидаемо, медианные оценки вкусовых качеств вина растут с увеличением 
# продолжительности его жизни. При этом интересно то, что медианное значение 
# вксуовых оценок вин без указания ограничений жизненного цикла, близки к 
# молодому вину.
  

# Теперь посмотрим как экспертами оценивались вина разных стран, а также как 
# соотносятся медианная цена и медианная стоимость вина.

# Создаем таблицу country_points
country_points <- df %>%
  group_by(country) %>%
  summarise(
    min = min(points, na.rm = TRUE),
    p25 = quantile(points, probs = 0.25, na.rm = TRUE),
    median_points = median(points, na.rm = TRUE),
    median_price = median(price, na.rm = TRUE),
    p75 = quantile(points, probs = 0.75, na.rm = TRUE),
    max = max(points, na.rm = TRUE),
    estim = median_price / (median_points),
    n = n()
  ) %>%
  filter(n > 50) %>%
  select(
    country, n, min, p25, median_points, p75, max, median_price, estim
  ) %>%
  arrange(desc(median_points), desc(median_price), desc(estim))

# Теперь печатаем результат
print(country_points, n = Inf)  
# Здесь мы видим нечто удивительное, по оценкам экспертов лидируют английские 
# вина, в то время как французские, итальянские и испанския вина занимают 7, 12 
# и 21 места. Однако, всё встаёт на свои места, когда мы смотрим на минимальное
# значение для вин Англии и видим что оно равно 89! То есть по всей видимости 
# дегустаторы заведомо отказывались от дегустации дешёвых английских вин и
# дегустировали только дорогие. И посмотрев на значение медианной цены за бутылку
# мы видим что мы были правы в своих подозрениях. 
# 
# Что же касается вышеуказанных вин, то по всей видимости дегустаторы настроены 
# к этим винам предвзято, и склонны занижать их оценки, ожидая за те же деньги
# более высоких вкусовых качеств. При этом, они отмечают австрийские, немецкие и
# канадские вина. Если мы посмотрим какие вина лучше всего оценены исходя из их
# бюджетности (цена/качество), мы видим что это вина Румынии, Молдавии и 
# Болгарии. Самое низкое медианное качество, по мнению дегустаторов, у 
# мексиканского и у молдавского вин.


# Теперь посмотрим как на оценки влияет сорт винограда.

analyze_scores <- function(data, group_col) {
  data %>%
    group_by(.data[[group_col]]) %>%  
    summarise(
      min = min(points, na.rm = TRUE),
      median = median(points, na.rm = TRUE),
      mean = round(mean(points, na.rm = TRUE), 1),
      max = max(points, na.rm = TRUE),
      sd = round(sd(points, na.rm = TRUE), 1),
      count = n(),
      median_price=median(price, na.rm = TRUE)
    ) %>%
    filter(count>5)%>%
    arrange(desc(median), desc(mean))
}

result <- analyze_scores(df, "variety")
print(head(result))
print(tail(result))
# У нас медианно лидирует сорт Bual, а хуже всего себя 
# зарекомендовали вина сорта Chambourcin.
# Посмотрим какова частота дегустации вин представленных этими сортами винограда

df %>%
  group_by(variety) %>%
  summarise(
    count = sum(!is.na(points)),
    percent = 100 * count / nrow(df), .groups = "drop") %>%
  arrange(desc(percent)) %>%
  head(5) 
  
pie_data <- df %>%
  filter(!is.na(points)) %>%
           count(variety, name = "count") %>%
           mutate(percent = 100 * count / sum(count)) %>%
           arrange(desc(count)) %>%
           head(20)
cat(pie_data[['variety']], sep=', ')
sum(pie_data$percent)


pie(pie_data$percent,
   labels = paste(pie_data$variety, " (", round(pie_data$percent, 1), "%)"),
   main = "Доля оценок по сортам вин (топ-20)",
   col = rainbow(nrow(pie_data)),
   cex = 0.8) 
  
# Тройка самых популярных вин: Пино Нуар, Шардоне и Каберне Савиньон.

# Выберем 20 сортов винограда и посмотрим, различимы ли популяции их рецензий
selected_varieties <- c("Pinot Noir", "Chardonnay", "Cabernet Sauvignon",
                        "Red Blend", "Bordeaux-style Red Blend", "Riesling",
                        "Sauvignon Blanc", "Syrah", "Rosé", "Merlot", "Nebbiolo", 
                        "Zinfandel", "Sangiovese", "Malbec", "Portuguese Red", 
                        "White Blend", "Sparkling Blend", "Tempranillo", 
                        "Rhône-style Red Blend", "Pinot Gris")




# Подготовка данных и Проверка условий для параметрических тестов

df_filtered <- df %>%
  filter(variety %in% selected_varieties) %>%
  mutate(variety = factor(variety)) %>%
  group_by(variety) %>%
  filter(n() >= 3) %>%  # Минимум 3 наблюдения в группе
  ungroup()

# Проверка нормальности (Шапиро-Уилк)
check_normality <- function(data) {
  data %>%
    group_by(variety) %>%
    summarise(
      shapiro_p = if (n() >= 3 && n() <= 5000) {
        shapiro.test(points)$p.value
      } else {
        NA_real_
      },
      .groups = "drop"
    )
}

shapiro_results <- check_normality(df_filtered)
print("Результаты теста Шапиро-Уилка:")
print(shapiro_results)
# Данные не проходят тест на нормальность

# 2. 2 Проверка гомогенности дисперсий (Левене или Бартлетт)
check_homogeneity <- function(data) {
  if (!requireNamespace("car", quietly = TRUE)) {
    install.packages("car")
  }
  car::leveneTest(points ~ variety, data = data)
}

homogeneity_test <- check_homogeneity(df_filtered)
print("Результаты теста на гомогенность дисперсий:")
print(homogeneity_test)

# Вариативность оценок существенно различается между сортами вин, данные нарушают
# условие однородности дисперсий, важное для ANOVA и t-тестов. Разные сорта имеют 
# неодинаковую "стабильность" качества: Некоторые сорта получают стабильно близкие
# оценки (малая дисперсия), другие — сильно варьирующиеся (большая дисперсия).

# Тест Краскела-Уоллиса
  print(kruskal.test(points ~ variety, data = df_filtered))
  
# Получен крайне значимый результат (p< 0.001): Мы отвергаем нулевую гипотезу, 
# существуют статистически значимые различия в медианных оценках сортов вин.

# Пост-хок тест Данна
if (!requireNamespace("FSA", quietly = TRUE)) {
  install.packages("FSA")
}
cat("\nПопарные сравнения (тест Данна с поправкой Бонферрони):\n")
dunn_result <- FSA::dunnTest(points ~ variety, 
                             data = df_filtered,
                             method = "bonferroni") 

# Извлечение результатов
dunn_df <- as.data.frame(dunn_result$res) 
sig_dunn <- dunn_df[dunn_df$P.adj>0.05,]

sig_dunn
  
# В результате попарных сравнений выявленны сочетания с высочайшими значениями 
# P-Value, свидетельсчтвующие о статистической неразличимости совокупностей вин
# сопоставляемых групп. После комбинирования, мы получае следующие группы вин:
# 1) "Red Blend", "Sangiovese", "Pinot Gris", "Chardonnay", "Cabernet Sauvignon",
# "Portuguese Red"}
# 2) "Rhône-style Red Blend", "Riesling", "Syrah", "Bordeaux-style Red Blend", 
  "Pinot Noir"
# 3) "Merlot", "Tempranillo", "White Blend", "Sauvignon Blanc"
# 4) "Malbec", "Sparkling Blend", "Zinfandel"
  
# Что объединяет эти вина? ChatGPT предлагает следующую интерпретацию:
# Группа 1: «классический ассортимент винной карты» — с географическим и 
# стилистическим разнообразием.
  
# Группа 2: "Вина Старого Света"/стили вин, ориентированные на традиционные 
# европейские регионы (особенно Франция).Европейские стили вин с выраженным 
# терруаром Сильная привязка к регионам Франции и Германии:
  
# Группа 3: "Вина на каждый день": сбалансированные, понятные, лёгкие для 
# восприятия, средней ценовой категории. Повседневные, легко доступные сорта и
# стили. средней ценовой категории
  
# Группа 4: "Экспрессивные, полнотелые или праздничные вина", Яркие, интенсивные,
# выразительные вина часто выбираемые для вечеринок или сильных гастрономических 
# акцентов.
  
# Следует отдельно изучить эти группы кластерным анализом.


# 5. Визуализация распределений оценок по сортам.

ggplot(df_filtered, aes(x = variety, y = points, fill = variety)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Распределение оценок по сортам вин",
       x = "Сорт вина",
       y = "Оценка (points)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Создаём бинарные признаки принадлежности ко группам

group1 <- c("Red Blend", "Sangiovese", "Pinot Gris", "Chardonnay", 
            "Cabernet Sauvignon", "Portuguese Red")
group2 <- c("Rhône-style Red Blend", "Riesling", "Syrah", 
            "Bordeaux-style Red Blend", "Pinot Noir")
group3 <- c("Merlot", "Tempranillo", "White Blend", "Sauvignon Blanc")
group4 <- c("Malbec", "Sparkling Blend", "Zinfandel")

# Функция для проверки вхождения любого элемента группы в строку
check_group <- function(variety, group) {
  sapply(variety, function(x) any(sapply(group, function(g) grepl(g, x, fixed = TRUE))))
}

df$classic <- as.integer(check_group(df$variety, group1))
df$franger <- as.integer(check_group(df$variety, group2))
df$everyday <- as.integer(check_group(df$variety, group3))
df$holiday <- as.integer(check_group(df$variety, group4)) 

# Проверяем результат находя количество наблюдений в каждой группе

sum(df$classic)
sum(df$franger)
sum(df$everyday)
sum(df$holiday)

################################################################################
################################################################################

# ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ ПО ПРОЕКТУ.


# Мы изучили данные, и судя по тому что мы видели, построение модели 
# предсказывающей оценку эксперта, представляется возможным. Далее мы можем
# осуществить импьютинг отсутствующих значений, конструирование новых признаков,
# а также извлечения эмбелдингов из описания, позволяющих базироваться в своих
# предсказаниях не только на численных признаках которые мы имеем и переменых 
# горячего кодирования, которые мы планируем создавать, но и работать с признаками
# оценки тональности отзыва. 
################################################################################
################################################################################
