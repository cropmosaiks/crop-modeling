##########################################
# ----------- MAP OF ZAMBIA --------------
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

country_vect <- country_shp %>%
  sf::st_union() %>%
  terra::vect()

zmb_union <- terra::vect(country_shp) %>%
  terra::buffer(0.1) %>%
  terra::aggregate()

africa_ext <- here::here('data', 'geo_boundaries', 'africa_ext.geojson') %>%
  terra::vect()

africa <- ne_countries(scale = "medium", returnclass = "sf") %>% 
  dplyr::filter(continent == 'Africa')  %>%
  terra::vect() |> 
  terra::crop(africa_ext)


extent <- terra::ext(zmb_union)
extent <- terra::as.polygons(extent)
terra::crs(extent) <- "epsg:4326"

provinces <- here::here('data', 'geo_boundaries', 'gadm36_ZMB_1.shp') %>% 
  sf::read_sf() %>% 
  dplyr::rename('province' = NAME_1)

crop_land <- here::here("data", "land_cover", "ZMB_cropland_2019.tif") %>%
  terra::rast() %>%
  terra::crop(country_vect, mask=TRUE) %>% 
  terra::aggregate(fact=33, fun='mean', na.rm = TRUE)

crop_land <- crop_land * 100

ybreaks <- c(-8, -10, -12, -14, -16, -18)
ylabs <- paste0(ybreaks,'°S')

xbreaks <- c(22, 24, 26, 28, 30, 32, 34)
xlabs <- paste0(xbreaks,'°E')

inset <- ggplot() +
  geom_sf(data = africa, fill = "white", linewidth = .5) +
  geom_sf(data = extent, fill = NA, linewidth = .5, color = 'red') +
  theme_void() +
  theme(panel.border = element_rect(fill = NA),
        panel.background = element_rect(fill = alpha("white", .5)))

main <- ggplot() +
  tidyterra::geom_spatraster(data = crop_land) +
  scale_fill_viridis_c(na.value = NA, guide = guide_colorbar(title.position = "top", barwidth = 7.5)) +
  geom_sf(data = country_shp, linewidth = .25, color = 'grey60', fill = NA) +
  geom_sf(data = provinces, linewidth = .5, color = 'black', fill = NA) +
  # geom_sf_label(data = provinces, aes(label = province) )+
  scale_size_identity() + 
  scale_x_continuous(breaks = xbreaks, labels = xlabs) +
  scale_y_continuous(breaks = ybreaks, labels = ylabs) +
  labs(x = NULL, y = NULL, fill = "Cropland Percentage") +
  theme_bw() +
  theme(legend.position = "bottom")

main_w_inset <- ggdraw() +
  draw_plot(main) +
  draw_plot(inset, x = 0.705, y = .178, width = .25, height = .25)

ggsave(
  filename = "figure_01.jpeg"
  , path = here("figures")
  , plot = main_w_inset
  , device ="jpeg"
  , width = 8
  , height = 7
  , units = "in"
)
