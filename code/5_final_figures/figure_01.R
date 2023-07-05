##########################################
######## FIGURE 1 - MAP OF ZAMBIA ######## 
##########################################

if (!require(librarian,quietly = T)){
  install.packages('librarian')
}

librarian::shelf(
  tidyverse,
  here,
  sf,
  terra,
  tidyterra,
  cowplot,
  rnaturalearth,
  rnaturalearthdata,
  quiet = T
)

country_shp <- here::here('data', 'geo_boundaries', 'gadm36_ZMB_2.shp') %>%
  sf::read_sf()

zmb_union <- terra::vect(country_shp) %>%
  terra::buffer(0.1) %>%
  terra::aggregate()

africa <- ne_countries(scale = "medium", returnclass = "sf") %>% 
  dplyr::filter(continent == 'Africa')

extent <- terra::ext(zmb_union)
extent <- terra::as.polygons(extent)
terra::crs(extent) <- "epsg:4326"

provinces <- here::here('data', 'geo_boundaries', 'gadm36_ZMB_1.shp') %>% 
  sf::read_sf() %>% 
  dplyr::rename('province' = NAME_1)

ybreaks <- c(-8, -10, -12, -14, -16, -18)
ylabs <- paste0(ybreaks,'°S')

xbreaks <- c(22, 24, 26, 28, 30, 32, 34)
xlabs <- paste0(xbreaks,'°E')

inset <- ggplot() +
  geom_sf(data = africa, fill = "white", linewidth = 1) +
  geom_sf(data = extent, fill = NA, linewidth = 1, color = 'red') +
  theme_void() +
  theme(panel.border = element_rect(fill = NA),
        panel.background = element_rect(fill = alpha("white", .5)))

main <- ggplot() +
  geom_sf(data = country_shp, linewidth = .5, color = 'black', fill = "grey60") +
  geom_sf(data = provinces, linewidth = 1, color = 'black', fill = NA) +
  geom_sf_label(data = provinces, aes(label = province) )+
  scale_fill_viridis_d() +
  scale_size_identity() + 
  scale_x_continuous(breaks = xbreaks, labels = xlabs) +
  scale_y_continuous(breaks = ybreaks, labels = ylabs) +
  labs(x = NULL, y = NULL) +
  theme_bw() +
  theme(legend.position = "right")

ggdraw() +
  draw_plot(main) +
  draw_plot(inset, x = 0.715, y = .065, width = .3, height = .3)

ggsave(
  filename = "figure_01.jpeg"
  , path = here("figures")
  , plot = ggplot2::last_plot()
  , device ="jpeg"
  , width = 8
  , height = 7
  , units = "in"
)
