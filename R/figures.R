# Publication figures, November (JCF): ggplot2 + patchwork. Input: results/figdata/* (scripts/16_figure_data.py) and results/*.json.
# Main: Figure 1 design + flow; Figure 2 next-year FEV1, observed vs predicted (carried forward vs machine learning) with 95%
# prediction intervals and the share within ±5/±10 points; Figure 3 the two exacerbation tools: points chart of the CFRT
# exacerbation score (A), calibration of score and machine learning (B), SHAP drivers of the machine-learning model (C).
# Supplement: S1 decision curves; S2 decline; S3–S4 SHAP full panels (shap red–blue palette); S5 leave-one-centre-out;
# S6 subgroups (children/adults); S7 calendar year.
suppressPackageStartupMessages({library(ggplot2); library(patchwork); library(dplyr); library(tidyr); library(readr); library(jsonlite); library(scales)})
here <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
FD <- file.path(here, "results", "figdata"); FG <- file.path(here, "results", "figures"); RS <- file.path(here, "results")
dir.create(FG, showWarnings = FALSE)
FONT <- "Arial"
NAVY <- "#1B4F9C"; CRIMSON <- "#C8102E"; AMBER <- "#F3A712"; TEAL <- "#17A398"; INK <- "grey12"; ORANGE <- "#E4572E"
COL <- c("CFRT exacerbation score" = NAVY, "Machine learning" = ORANGE, "IV-antibiotic days alone" = TEAL, "LR-5 (severity index)" = "#7F8C8D",
         "Rule: any IV (t−1/t)" = "#8E44AD", "Rule: severity index ≥2" = AMBER, "Rule: severity index (FEV1 <80%) ≥2" = "#B8860B",
         "Treat all" = "grey55", "Treat none" = "grey20")
LTY <- c("CFRT exacerbation score" = 1, "Machine learning" = 1, "IV-antibiotic days alone" = 1, "LR-5 (severity index)" = 1, "Rule: any IV (t−1/t)" = 2,
         "Rule: severity index ≥2" = 2, "Rule: severity index (FEV1 <80%) ≥2" = 2, "Treat all" = 3, "Treat none" = 1)
LWD <- c("CFRT exacerbation score" = 1.35, "Machine learning" = 0.75, "IV-antibiotic days alone" = 0.75, "LR-5 (severity index)" = 0.6, "Rule: any IV (t−1/t)" = 0.75,
         "Rule: severity index ≥2" = 0.75, "Rule: severity index (FEV1 <80%) ≥2" = 0.75, "Treat all" = 0.6, "Treat none" = 0.5)
DISP <- c("CFRT exacerbation score" = "CFRT exacerbation score", "Machine learning" = "Machine learning", "IV-antibiotic days alone" = "IV-antibiotic days alone",
          "LR-5 (severity index)" = "LR-5 (severity index)", "Rule: any IV (t−1/t)" = "Rule: any IV in t−1/t", "Rule: severity index ≥2" = "Rule: severity index ≥2",
          "Rule: severity index (FEV1 <80%) ≥2" = "Rule: severity ≥2 (FEV1 <80%)", "Treat all" = "Treat all", "Treat none" = "Treat none")
SHAP_PAL <- c("#008bfb", "#007df5", "#236cea", "#625adb", "#8443c6", "#9c23ad", "#bb00a0", "#d5008f", "#e9007c", "#f90067", "#ff0051")  # shap.plots.colors.red_blue
th <- theme_classic(base_size = 9, base_family = FONT) +
  theme(legend.title = element_blank(), plot.title = element_text(face = "bold", size = 9.5, colour = INK), plot.subtitle = element_text(size = 7.5, colour = "grey35"),
        axis.text = element_text(colour = "grey15"), axis.title = element_text(colour = INK), axis.line = element_line(colour = "grey30", linewidth = 0.4),
        axis.ticks = element_line(colour = "grey30", linewidth = 0.4), plot.tag = element_text(face = "bold", size = 13, colour = INK),
        panel.grid.major.y = element_line(colour = "grey93", linewidth = 0.3))
save <- function(p, name, w, h) {
  ggsave(file.path(FG, paste0(name, ".png")), p, width = w, height = h, dpi = 300, device = ragg::agg_png)
  ggsave(file.path(FG, paste0(name, ".pdf")), p, width = w, height = h, device = cairo_pdf) }
fmt2 <- function(x) gsub("-", "−", sub("^-0\\.00$", "0.00", sprintf("%.2f", x)))
wilson <- function(x, n) { z <- 1.96; p <- x / n; d <- 1 + z^2 / n
  c((p + z^2 / (2 * n) - z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / d, (p + z^2 / (2 * n) + z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / d) }
spread <- function(y, gap, lo = -Inf) { o <- order(y); y0 <- y[o]; ys <- y0; for (i in seq_along(ys)[-1]) if (ys[i] - ys[i - 1] < gap) ys[i] <- ys[i - 1] + gap
  ys <- ys - mean(ys - y0); ys <- ys + max(0, lo - min(ys)); y2 <- y; y2[o] <- ys; y2 }
cm <- function(x) comma(x)

# ---------------- calibration and decision-curve building blocks
calib_main <- function(outcome_, models = c("CFRT exacerbation score", "Machine learning"), lim = 0.8) {
  rows <- read_csv(file.path(FD, "calibration_rows.csv"), show_col_types = FALSE) |> filter(outcome == outcome_, model %in% models) |> mutate(model = factor(model, levels = models))
  st <- read_csv(file.path(FD, "calibration_stats.csv"), show_col_types = FALSE) |> filter(outcome == outcome_, model %in% models) |>
    mutate(model = factor(model, levels = models)) |> arrange(model) |>
    mutate(l1 = sprintf("CITL %s (%s to %s)", fmt2(citl), fmt2(citl_lo), fmt2(citl_hi)), l2 = sprintf("Slope %s (%s–%s)", fmt2(slope), fmt2(slope_lo), fmt2(slope_hi)))
  dec <- rows |> group_by(model) |> mutate(bin = ntile(p, 10)) |> group_by(model, bin) |> summarise(p = mean(p), n = n(), x = sum(y), .groups = "drop") |>
    rowwise() |> mutate(obs = x / n, lo = wilson(x, n)[1], hi = wilson(x, n)[2]) |> ungroup()
  k <- nrow(st); step <- 0.155 * lim; top <- 0.03 * lim + k * step; yy <- top - step * (seq_len(k) - 1) - 0.012 * lim; bx <- 0.50 * lim
  ggplot(rows, aes(p, y, colour = model, fill = model)) +
    annotate("rect", xmin = bx - 0.02 * lim, xmax = 0.985 * lim, ymin = 0.02 * lim, ymax = top + 0.005 * lim, fill = "white", colour = "grey80", linewidth = 0.3) +
    geom_abline(linetype = 2, colour = "grey55", linewidth = 0.4) +
    geom_smooth(aes(linewidth = model), method = "loess", formula = y ~ x, span = 0.9, se = TRUE, alpha = 0.13) +
    geom_linerange(data = dec, aes(p, ymin = lo, ymax = hi), linewidth = 0.4, inherit.aes = FALSE, colour = COL[as.character(dec$model)]) +
    geom_point(data = dec, aes(p, obs, fill = model), shape = 21, colour = "white", size = 2, stroke = 0.4) +
    annotate("text", x = bx, y = yy, hjust = 0, vjust = 1, size = 2.6, family = FONT, fontface = "bold", label = as.character(st$model), colour = COL[as.character(st$model)]) +
    annotate("text", x = bx, y = yy - 0.05 * lim, hjust = 0, vjust = 1, size = 2.25, family = FONT, label = st$l1, colour = "grey20") +
    annotate("text", x = bx, y = yy - 0.095 * lim, hjust = 0, vjust = 1, size = 2.25, family = FONT, label = st$l2, colour = "grey20") +
    scale_colour_manual(values = COL) + scale_fill_manual(values = COL) + scale_linewidth_manual(values = LWD) +
    coord_cartesian(xlim = c(0, lim), ylim = c(0, lim), expand = FALSE) + scale_x_continuous(labels = NULL) +
    labs(x = NULL, y = "Observed proportion") + th + theme(legend.position = "none", plot.margin = margin(4, 8, 0, 4))
}
calib_hist <- function(outcome_, model_ = "CFRT exacerbation score", lim = 0.8, bw = 0.02) {
  r <- read_csv(file.path(FD, "calibration_rows.csv"), show_col_types = FALSE) |> filter(outcome == outcome_, model == model_, p <= lim) |>
    mutate(bin = floor(p / bw) * bw + bw / 2) |> count(bin, y) |> mutate(n = ifelse(y == 1, n, -n), grp = ifelse(y == 1, "Event", "No event"))
  mxn <- max(-r$n[r$n < 0], 1); up <- max(max(r$n[r$n > 0], 1) * 1.15, mxn * 0.3)
  ggplot(r, aes(bin, n, fill = grp)) + geom_col(width = bw * 0.92) + geom_hline(yintercept = 0, colour = "grey30", linewidth = 0.3) +
    annotate("text", x = lim * 0.985, y = up * 0.55, hjust = 1, size = 2.2, family = FONT, label = "with event", colour = COL[model_], fontface = "bold") +
    annotate("text", x = lim * 0.985, y = -mxn * 0.55, hjust = 1, size = 2.2, family = FONT, label = "without event", colour = "grey40") +
    scale_fill_manual(values = c("Event" = unname(COL[model_]), "No event" = "grey72")) + scale_y_continuous(labels = function(v) abs(v), breaks = pretty_breaks(3)) +
    coord_cartesian(xlim = c(0, lim), ylim = c(-mxn * 1.04, up), expand = FALSE) + labs(x = sprintf("Predicted probability (%s)", model_), y = "Prediction-years") + th +
    theme(legend.position = "none", panel.grid.major.y = element_blank(), plot.margin = margin(0, 8, 4, 4))
}
dca_main <- function(outcome_, strategies = c("CFRT exacerbation score", "Machine learning", "IV-antibiotic days alone", "Rule: any IV (t−1/t)", "Rule: severity index ≥2", "Treat all"),
                     shade = TRUE, gap = 0.0115) {
  d <- read_csv(file.path(FD, "dca.csv"), show_col_types = FALSE) |> filter(outcome == outcome_, strategy %in% strategies, threshold <= 0.5) |>
    mutate(strategy = factor(strategy, levels = c(strategies, "Treat none")))
  prev <- max(d$nb[d$strategy == "Treat all"], na.rm = TRUE); ymin <- -0.03
  lab <- d |> filter(strategy != "Treat all") |> group_by(strategy) |> filter(threshold == max(threshold)) |> ungroup() |> select(strategy, nb) |>
    bind_rows(tibble(strategy = factor("Treat none", levels = levels(d$strategy)), nb = 0)) |> mutate(y = spread(nb, gap, lo = ymin + 0.004))
  ta <- d |> filter(strategy == "Treat all", nb >= ymin) |> slice_min(abs(nb - ymin / 2), n = 1)
  p <- ggplot(d, aes(threshold, nb, colour = strategy, linetype = strategy, linewidth = strategy))
  if (shade) p <- p + annotate("rect", xmin = 0.15, xmax = 0.30, ymin = -Inf, ymax = Inf, fill = "#FFF1D0", alpha = 0.8) +
    annotate("text", x = 0.225, y = prev + 0.012, label = "Thresholds 0.15–0.30", size = 2.3, family = FONT, colour = "#9A6A00", fontface = "italic")
  p + geom_hline(yintercept = 0, colour = COL["Treat none"], linewidth = 0.5) + geom_line(data = d |> filter(nb >= ymin)) +
    geom_segment(data = lab, aes(x = 0.5, xend = 0.512, y = nb, yend = y, colour = strategy), linewidth = 0.3, linetype = 1, inherit.aes = FALSE) +
    geom_text(data = lab, aes(x = 0.515, y = y, label = DISP[as.character(strategy)], colour = strategy, fontface = ifelse(strategy == "CFRT exacerbation score", "bold", "plain")),
              hjust = 0, size = 2.4, family = FONT, inherit.aes = FALSE) +
    annotate("text", x = ta$threshold + 0.008, y = ta$nb, label = "Treat all", hjust = 0, size = 2.3, family = FONT, colour = COL["Treat all"]) +
    scale_colour_manual(values = COL) + scale_linetype_manual(values = LTY) + scale_linewidth_manual(values = LWD) +
    coord_cartesian(ylim = c(ymin, prev + 0.02), xlim = c(0.05, 0.5), expand = FALSE, clip = "off") + scale_x_continuous(breaks = seq(0.1, 0.5, 0.1)) +
    labs(x = "Threshold probability", y = "Net benefit per prediction-year") + th + theme(legend.position = "none", plot.margin = margin(4, 92, 4, 4))
}

# ---------------- Figure 1: study design (A) and cohort flow (B)
fl <- fromJSON(file.path(FD, "flow.json"))
blk <- tibble(x0 = c(0, 1.04, 2.08), x1 = c(1, 2.04, 3.08), fill = c("#DCE6F5", NAVY, "#FCE3DA"), tc = c(INK, "white", "#8A1C0E"),
              head = c("Year t−1", "Anchor year t", "Outcome year t+1"),
              body = c("History: FEV1, growth,\nIV-antibiotic days", "Year-end review\nFEV1 z · BMI z · cultures · IV days", "FEV1 (z, % predicted) · ≥10-point fall\nIV-treated exacerbation"))
fA <- ggplot() +
  geom_rect(data = blk, aes(xmin = x0, xmax = x1, ymin = 0.55, ymax = 1.55, fill = I(fill))) +
  geom_text(data = blk, aes(x = (x0 + x1) / 2, y = 1.32, label = head, colour = I(tc)), fontface = "bold", size = 3.1, family = FONT) +
  geom_text(data = blk, aes(x = (x0 + x1) / 2, y = 0.92, label = body, colour = I(tc)), size = 2.45, family = FONT, lineheight = 0.95) +
  annotate("segment", x = 2.06, xend = 2.06, y = 0.45, yend = 1.78, colour = CRIMSON, linewidth = 0.9) +
  annotate("point", x = 2.06, y = 1.86, shape = 25, size = 2.6, fill = CRIMSON, colour = CRIMSON) +
  annotate("text", x = 2.06, y = 2.05, label = "Prediction made here", colour = CRIMSON, fontface = "bold", size = 2.7, family = FONT) +
  annotate("segment", x = c(0, 2.08), xend = c(2.04, 3.08), y = 0.38, yend = 0.38, colour = c(NAVY, "#C0392B"), linewidth = 0.6,
           arrow = arrow(ends = "both", length = unit(1.4, "mm"), type = "closed")) +
  annotate("text", x = c(1.02, 2.58), y = 0.2, label = c("Two years of registry data", "One-year horizon"), colour = c(NAVY, "#C0392B"), size = 2.45, family = FONT) +
  annotate("text", x = 1.54, y = -0.08, size = 2.35, family = FONT, colour = "grey35",
           label = "No CFTR modulator in t−1, t or t+1 · anchor years 2020–2023 · a patient contributes up to four prediction-years") +
  coord_cartesian(xlim = c(-0.05, 3.13), ylim = c(-0.2, 2.18), expand = FALSE) + theme_void(base_family = FONT) + labs(tag = "A") +
  theme(plot.tag = element_text(face = "bold", size = 13), plot.tag.position = c(0.01, 0.98))
st <- setNames(fl$exclusion_steps$excluded_rows, fl$exclusion_steps$step); rem <- fl$exclusion_steps
mainb <- tibble(y = c(5, 4, 3, 2),
                head = c("CFRT annual records with spirometry, 2019–2024", "Candidate anchor years 2020–2023", "Age >5 years with spirometry in t−1", "No CFTR modulator in t−1 or t"),
                sub = c("Study export (centre-reported annual review)", "Years t−1 and t+1 within 2019–2024", "Two consecutive annual records", "Modulator-untreated at the annual review"),
                num = c(sprintf("%s patient-years · %s patients", cm(fl$export_rows), cm(fl$export_patients)),
                        sprintf("%s patient-years · %s patients", cm(fl$candidate_anchor_rows), cm(fl$candidate_anchor_patients)),
                        sprintf("%s patient-years · %s patients", cm(rem$remaining_rows[2]), cm(rem$remaining_patients[2])),
                        sprintf("%s patient-years · %s patients", cm(rem$remaining_rows[3]), cm(rem$remaining_patients[3]))))
excb <- tibble(y = c(4.5, 3.5, 2.5, 1.58),
               txt = c(sprintf("Anchor year outside 2020–2023: %s", cm(fl$export_rows - fl$candidate_anchor_rows)),
                       sprintf("Age ≤5 years: %s\nNo spirometry record in t−1: %s", cm(st[["age_le_5_at_anchor"]]), cm(st[["no_spirometry_record_in_history_year"]])),
                       sprintf("Modulator in t−1 or t: %s", cm(st[["modulator_in_history_or_anchor_year"]])),
                       sprintf("No spirometry record in t+1: %s\nModulator in t+1: %s", cm(st[["no_spirometry_record_in_outcome_year"]]),
                               cm(st[["modulator_in_outcome_year"]]))))
hb <- 0.33; xl <- 0.04; xr <- 0.56; ex0 <- 0.645; ex1 <- 0.995; yF <- 0.78; hF <- 0.52; he <- 0.24
fB <- ggplot() +
  geom_segment(aes(x = 0.30, xend = 0.30, y = c(5, 4, 3) - hb, yend = c(4, 3, 2) + hb), colour = "grey35", linewidth = 0.6, arrow = arrow(length = unit(2, "mm"), type = "closed")) +
  geom_segment(aes(x = 0.30, xend = 0.30, y = 2 - hb, yend = yF + hF), colour = "grey35", linewidth = 0.6, arrow = arrow(length = unit(2, "mm"), type = "closed")) +
  geom_segment(data = excb, aes(x = 0.30, xend = ex0 - 0.005, y = y, yend = y), colour = "#C0392B", linewidth = 0.45, arrow = arrow(length = unit(1.6, "mm"), type = "closed")) +
  geom_rect(data = mainb, aes(xmin = xl, xmax = xr, ymin = y - hb, ymax = y + hb), fill = "#F2F6FC", colour = "#C9D6EA", linewidth = 0.3) +
  geom_rect(data = mainb, aes(xmin = xl, xmax = xl + 0.012, ymin = y - hb, ymax = y + hb), fill = NAVY) +
  geom_text(data = mainb, aes(x = xl + 0.03, y = y + 0.17, label = head), hjust = 0, fontface = "bold", size = 2.85, family = FONT, colour = INK) +
  geom_text(data = mainb, aes(x = xl + 0.03, y = y + 0.0, label = sub), hjust = 0, size = 2.35, family = FONT, colour = "grey35") +
  geom_text(data = mainb, aes(x = xl + 0.03, y = y - 0.18, label = num), hjust = 0, fontface = "bold", size = 2.9, family = FONT, colour = NAVY) +
  geom_rect(data = excb, aes(xmin = ex0, xmax = ex1, ymin = y - he, ymax = y + he), fill = "#FDEEEC", colour = "#F2C4BD", linewidth = 0.3) +
  geom_rect(data = excb, aes(xmin = ex0, xmax = ex0 + 0.009, ymin = y - he, ymax = y + he), fill = "#C0392B") +
  geom_text(data = excb, aes(x = ex0 + 0.025, y = y, label = txt), hjust = 0, size = 2.3, family = FONT, colour = "#5A1A12", lineheight = 0.95) +
  annotate("rect", xmin = xl, xmax = xr + 0.045, ymin = yF - hF, ymax = yF + hF, fill = NAVY) +
  annotate("text", x = xl + 0.03, y = yF + 0.33, hjust = 0, label = "Analysis cohort: modulator-untreated in t−1, t and t+1", fontface = "bold", size = 2.9, family = FONT, colour = "#DCE6F5") +
  annotate("text", x = xl + 0.03, y = yF + 0.10, hjust = 0, label = sprintf("%s prediction-years · %s patients · %s centres", cm(fl$final_rows), fl$final_patients, fl$centres),
           fontface = "bold", size = 3.5, family = FONT, colour = "white") +
  annotate("text", x = xl + 0.03, y = yF - 0.14, hjust = 0, size = 2.6, family = FONT, colour = "white",
           label = sprintf("IV-treated exacerbation %s/%s (%.1f%%) · ≥10-point FEV1 fall %s/%s", fl$events_y_pex, cm(fl$n_y_pex), 100 * fl$events_y_pex / fl$n_y_pex,
                           fl$events_y_decline10, cm(fl$n_y_decline10))) +
  annotate("text", x = xl + 0.03, y = yF - 0.34, hjust = 0, size = 2.35, family = FONT, colour = "#DCE6F5",
           label = sprintf("Adults (≥18 years in the anchor year): %s prediction-years · age range %s–%s years", fl$adult_rows, fl$anchor_age_min_max[1], fl$anchor_age_min_max[2])) +
  coord_cartesian(xlim = c(0, 1), ylim = c(yF - hF - 0.05, 5.4), expand = FALSE) + theme_void(base_family = FONT) + labs(tag = "B") +
  theme(plot.tag = element_text(face = "bold", size = 13), plot.tag.position = c(0.01, 0.99))
save(fA / fB + plot_layout(heights = c(1.45, 5.0)), "Figure1_design_flow", 7.2, 7.6)

# ---------------- Figure 2: next-year FEV1 % predicted, observed vs predicted
fr <- read_csv(file.path(FD, "fev1_rows.csv"), show_col_types = FALSE)
fs <- read_csv(file.path(FD, "fev1_summary.csv"), show_col_types = FALSE)
scatter_fev1 <- function(tgt, key_, ttl, col, lims = c(10, 140)) {
  s <- fs |> filter(target == tgt, key == key_); d <- fr |> filter(target == tgt, key == key_) |> mutate(e = abs(y - pred),
    cls = case_when(e <= s$tol1 ~ "≤5", e <= s$tol2 ~ "5–10", TRUE ~ ">10"))
  hw <- s$pi_hw; t1 <- s$tol1; t2 <- s$tol2
  band <- tibble(x = lims, lo2 = lims - t2, hi2 = lims + t2, lo1 = lims - t1, hi1 = lims + t1, lop = lims - hw, hip = lims + hw)
  lab <- c(sprintf("Within ±%g points: %.0f%%", t1, 100 * s$w1), sprintf("Within ±%g points: %.0f%%", t2, 100 * s$w2),
           sprintf("95%% prediction interval: ±%.0f points", hw), sprintf("R² %.2f · mean error %.1f points", s$r2, s$mae))
  ggplot(d, aes(pred, y)) +
    geom_ribbon(data = band, aes(x = x, ymin = lop, ymax = hip), inherit.aes = FALSE, fill = "grey95") +
    geom_ribbon(data = band, aes(x = x, ymin = lo2, ymax = hi2), inherit.aes = FALSE, fill = alpha(col, 0.13)) +
    geom_ribbon(data = band, aes(x = x, ymin = lo1, ymax = hi1), inherit.aes = FALSE, fill = alpha(col, 0.22)) +
    geom_line(data = band, aes(x = x, y = lop), inherit.aes = FALSE, linetype = 2, colour = "grey45", linewidth = 0.35) +
    geom_line(data = band, aes(x = x, y = hip), inherit.aes = FALSE, linetype = 2, colour = "grey45", linewidth = 0.35) +
    geom_abline(colour = col, linewidth = 0.6) +
    geom_point(aes(fill = cls), shape = 21, colour = "white", stroke = 0.15, size = 1.25, alpha = 0.9) +
    scale_fill_manual(values = c("≤5" = col, "5–10" = alpha(col, 0.55), ">10" = "grey55"), guide = "none") +
    annotate("label", x = lims[1] + 2, y = lims[2] - 2, hjust = 0, vjust = 1, size = 2.45, family = FONT, fill = "white",
             label = paste(lab, collapse = "\n"), lineheight = 1.05, colour = INK) +
    annotate("text", x = lims[2] - 4, y = lims[2] - 4 - hw, label = "95% prediction interval", angle = 45, hjust = 1, vjust = 1.4, size = 2.1, family = FONT, colour = "grey30") +
    coord_equal(xlim = lims, ylim = lims, expand = FALSE) +
    labs(x = "Predicted next-year FEV1, % predicted", y = "Observed next-year FEV1, % predicted", title = ttl) + th
}
p2a <- scatter_fev1("y_fev1_pp", "persistence", "This year's value carried forward", "grey30") + labs(tag = "A")
p2b <- scatter_fev1("y_fev1_pp", "ml_change", "Machine learning", NAVY) + labs(tag = "B")
ord <- c("This year's value carried forward", "Machine learning")
sh <- fs |> filter(target == "y_fev1_pp", model %in% ord) |> select(model, share_le1, share_1_2, share_gt2) |>
  mutate(model = factor(model, levels = rev(ord)), x1 = share_le1 / 2, x2 = share_le1 + share_1_2 / 2, x3 = share_le1 + share_1_2 + share_gt2 / 2)
shl <- sh |> pivot_longer(starts_with("share"), names_to = "cat", values_to = "v") |>
  mutate(cat = factor(recode(cat, share_le1 = "Within 5 points", share_1_2 = "5–10 points off", share_gt2 = "More than 10 points off"),
                      levels = c("More than 10 points off", "5–10 points off", "Within 5 points")))
p2c <- ggplot(shl, aes(v, model, fill = cat)) + geom_col(width = 0.66, colour = "white", linewidth = 0.3) +
  geom_text(data = sh, aes(x = x1, y = model, label = sprintf("%.0f%%", 100 * share_le1)), inherit.aes = FALSE, colour = "white", size = 2.5, family = FONT, fontface = "bold") +
  geom_text(data = sh, aes(x = x2, y = model, label = sprintf("%.0f%%", 100 * share_1_2)), inherit.aes = FALSE, colour = NAVY, size = 2.4, family = FONT) +
  geom_text(data = sh, aes(x = x3, y = model, label = sprintf("%.0f%%", 100 * share_gt2)), inherit.aes = FALSE, colour = "grey30", size = 2.4, family = FONT) +
  scale_fill_manual(values = c("Within 5 points" = NAVY, "5–10 points off" = "#9DB7E0", "More than 10 points off" = "grey85"), breaks = c("Within 5 points", "5–10 points off", "More than 10 points off")) +
  scale_x_continuous(labels = percent, expand = c(0, 0)) + labs(x = "Share of prediction-years (out-of-fold predictions)", y = NULL, title = "How far was the prediction from next year's observed FEV1?", tag = "C") +
  th + theme(legend.position = "top", legend.justification = "left", legend.key.size = unit(3, "mm"), legend.text = element_text(size = 7.2), panel.grid.major.y = element_blank())
save(wrap_elements(full = p2a | p2b) / wrap_elements(full = p2c) + plot_layout(heights = c(2.2, 0.75)), "Figure2_fev1_observed_predicted", 7.6, 6.2)

# ---------------- Supplementary Figure S1: decision curves for IV-treated exacerbation
save(dca_main("y_pex") + labs(title = "Decision curves: IV-treated exacerbation in the next year"), "FigureS1_decision_curves", 5.6, 4.0)

# ---------------- points chart of the CFRT exacerbation score (Figure 3A)
nm <- fromJSON(file.path(FD, "nomogram.json"), simplifyVector = FALSE); vars <- nm$variables; sc <- nm$scale_points_per_logit; lp0 <- nm$linear_predictor_at_zero_points
SHORT <- c(FEV1_Z_lag_1 = "FEV1 z-score, anchor year", bmiZscore_lag_1 = "BMI z-score, anchor year", PseudomAer_lag_1 = "P. aeruginosa, anchor year",
           pa_any_2y = "P. aeruginosa, t−1 or t", MRSA_lag_1 = "MRSA, anchor year", mrsa_any_2y = "MRSA, t−1 or t", ABPA_lag_1 = "ABPA, anchor year", log_iv_burden_2yr = "IV-antibiotic days, t−1 + t", iv_burden_2yr_lag = "IV-antibiotic days, t−1 + t")
rowsN <- names(vars); nr <- length(rowsN) + 3; segs <- list(); tks <- list(); exm <- list()
for (i in seq_along(rowsN)) { v <- vars[[rowsN[i]]]; yy <- nr - i
  tv <- names(v$points_at_ticks); pts <- unlist(v$points_at_ticks)
  lab_ <- if (rowsN[i] %in% c("PseudomAer_lag_1", "pa_any_2y", "MRSA_lag_1", "mrsa_any_2y", "ABPA_lag_1", "abpa_any_2y")) ifelse(as.numeric(tv) == 0, "No", "Yes") else gsub("-", "−", tv)
  segs[[i]] <- tibble(y = yy, x0 = min(pts), x1 = max(pts), lab = SHORT[[rowsN[i]]]); tks[[i]] <- tibble(y = yy, x = pts, txt = lab_)
  exm[[i]] <- tibble(y = yy, x = nm$example_points[[rowsN[i]]]) }
segs <- bind_rows(segs); tks <- bind_rows(tks); exm <- bind_rows(exm) |> mutate(txt = sprintf("%.0f", x))
maxtot <- ceiling(sum(segs$x1 - pmin(segs$x0, 0)) / 20) * 20; xT <- function(tot) tot * 100 / maxtot
tot_ex <- sum(exm$x); risk_ex <- plogis(lp0 + tot_ex / sc)
risk_t <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9); tot_t <- (qlogis(risk_t) - lp0) * sc
risk_df <- tibble(x = xT(tot_t), r = risk_t) |> filter(x >= 0, x <= 100) |> mutate(txt = sprintf("%.2f", r))
tot_df <- tibble(x = xT(seq(0, maxtot, 20)), txt = seq(0, maxtot, 20))
yP <- nr; yT <- 1.75; yR <- 0.7
grad <- tibble(r = seq(min(risk_df$r), max(risk_df$r), length.out = 400)) |> mutate(x = xT((qlogis(r) - lp0) * sc), x2 = lead(x, default = max(x)) + 0.15)
band <- xT((qlogis(c(0.15, 0.30)) - lp0) * sc)
zebra <- tibble(y = c(yP, segs$y, yT, yR)) |> mutate(k = row_number()) |> filter(k %% 2 == 1)
fS1 <- ggplot() +
  geom_rect(data = zebra, aes(xmin = -78, xmax = 106, ymin = y - 0.48, ymax = y + 0.48), fill = "#F3F6FB") +
  annotate("rect", xmin = band[1], xmax = band[2], ymin = yR - 0.42, ymax = yR + 0.5, fill = "#FFF1D0") +
  annotate("text", x = mean(band), y = yR - 0.3, label = "0.15–0.30", size = 2.2, family = FONT, fontface = "italic", colour = "#9A6A00") +
  annotate("segment", x = 0, xend = 100, y = yP, yend = yP, colour = "grey25", linewidth = 0.6) +
  geom_segment(data = tibble(x = seq(0, 100, 10)), aes(x = x, xend = x, y = yP, yend = yP + 0.14), colour = "grey25", linewidth = 0.4) +
  geom_text(data = tibble(x = seq(0, 100, 10)), aes(x = x, y = yP + 0.33, label = x), size = 2.4, family = FONT, colour = "grey15") +
  geom_segment(data = segs, aes(x = x0, xend = x1, y = y, yend = y), colour = NAVY, linewidth = 1.1, lineend = "round") +
  geom_segment(data = tks, aes(x = x, xend = x, y = y, yend = y + 0.14), colour = NAVY, linewidth = 0.45) +
  geom_text(data = tks, aes(x = x, y = y + 0.33, label = txt), size = 2.35, family = FONT, colour = INK) +
  geom_point(data = exm, aes(x, y), shape = 21, fill = CRIMSON, colour = "white", size = 2.6, stroke = 0.6) +
  geom_text(data = exm, aes(x, y - 0.27, label = paste0(txt, " pts")), size = 2.1, family = FONT, colour = CRIMSON, fontface = "bold") +
  annotate("segment", x = 0, xend = 100, y = yT, yend = yT, colour = "grey25", linewidth = 0.6) +
  geom_segment(data = tot_df, aes(x = x, xend = x, y = yT, yend = yT + 0.14), colour = "grey25", linewidth = 0.4) +
  geom_text(data = tot_df, aes(x = x, y = yT + 0.33, label = txt), size = 2.35, family = FONT, colour = "grey15") +
  geom_rect(data = grad, aes(xmin = x, xmax = x2, ymin = yR - 0.09, ymax = yR + 0.09, fill = r), colour = NA) +
  scale_fill_gradientn(colours = c("#2E9E5B", "#A7C957", AMBER, "#E4572E", "#A4161A"), values = rescale(c(0.05, 0.15, 0.25, 0.45, 0.9)), guide = "none") +
  geom_segment(data = risk_df, aes(x = x, xend = x, y = yR + 0.1, yend = yR + 0.2), colour = "grey25", linewidth = 0.4) +
  geom_text(data = risk_df, aes(x = x, y = yR + 0.36, label = txt), size = 2.35, family = FONT, colour = "grey15") +
  annotate("segment", x = xT(tot_ex), xend = xT(tot_ex), y = yT, yend = yR + 0.08, colour = CRIMSON, linewidth = 0.5, linetype = 2) +
  annotate("point", x = xT(tot_ex), y = yT, shape = 21, fill = CRIMSON, colour = "white", size = 2.8, stroke = 0.6) +
  annotate("point", x = xT(tot_ex), y = yR, shape = 23, fill = "white", colour = CRIMSON, size = 3.2, stroke = 1) +
  annotate("text", x = xT(tot_ex) + 1.2, y = yT - 0.27, hjust = 0, size = 2.2, family = FONT, colour = CRIMSON, fontface = "bold", label = sprintf("Total %.0f points", tot_ex)) +
  annotate("text", x = xT(tot_ex) + 1.8, y = yR - 0.3, hjust = 0, size = 2.3, family = FONT, colour = CRIMSON, fontface = "bold", label = sprintf("Predicted risk %.2f", risk_ex)) +
  geom_text(data = bind_rows(tibble(y = yP, lab = "Points"), segs |> select(y, lab), tibble(y = c(yT, yR), lab = c("Total points", "One-year risk"))),
            aes(x = -4, y = y + 0.05, label = lab), hjust = 1, size = 2.75, family = FONT, fontface = "bold", colour = INK) +
  annotate("text", x = -4, y = yR - 0.24, hjust = 1, size = 2.2, family = FONT, colour = "grey35", label = "IV-treated exacerbation") +
  coord_cartesian(xlim = c(-78, 106), ylim = c(yR - 0.55, yP + 0.55), expand = FALSE, clip = "off") + theme_void(base_family = FONT) +
  theme(plot.margin = margin(6, 6, 4, 6))

# ---------------- Figure S2: decline
fs2 <- ((calib_main("y_decline10", lim = 0.6) + labs(title = "Calibration, ≥10-point fall", tag = "A")) / calib_hist("y_decline10", lim = 0.6) + plot_layout(heights = c(3.3, 1))) |
  (dca_main("y_decline10", c("CFRT exacerbation score", "Machine learning", "IV-antibiotic days alone", "Treat all"), shade = FALSE) + labs(title = "Decision curves", tag = "B"))
save(fs2, "FigureS2_decline", 7.6, 3.9)

# ---------------- SHAP (shap red–blue palette and beeswarm conventions)
NICE <- c(FEV1_Z_lag_1 = "FEV1 z, anchor", FEV1_Z_lag_2 = "FEV1 z, t−1", iv_burden_2yr_lag = "IV days, t−1 + t", log_iv_burden_2yr = "IV days, t−1 + t (log)",
  ivDaysTotal_lag_1 = "IV days, anchor", ivDaysTotal_lag_2 = "IV days, t−1", severity_index_lag_1 = "Severity index", AgeAtVisit_lag_1 = "Age", bmiZscore_lag_1 = "BMI z, anchor",
  bmiZscore_lag_2 = "BMI z, t−1", PseudomAer_lag_1 = "P. aeruginosa, anchor", AgeDiagnosis = "Age at diagnosis", SweatTestVal = "Sweat chloride",
  FVC_Z_lag_1 = "FVC z, anchor", FVC_Z_lag_2 = "FVC z, t−1", PExNumber_lag_1 = "Exacerbations, anchor", PExNumber_lag_2 = "Exacerbations, t−1",
  fev1_pp_trend_lag = "FEV1 change t−1→t", fev1_z_trend_lag = "FEV1 z change t−1→t", fev1_z_mean2 = "FEV1 z, two-year mean", fev1_pp_mean2 = "FEV1 %, two-year mean",
  fvc_z_trend_lag = "FVC z change t−1→t", fev1_fvc_z_gap = "FEV1 z − FVC z", bmi_trend_lag = "BMI z change t−1→t", height_z_trend_lag = "Height z change",
  heightZScore_lag_1 = "Height z, anchor", heightZScore_lag_2 = "Height z, t−1", weightZScore_lag_1 = "Weight z, anchor", weightZScore_lag_2 = "Weight z, t−1",
  high_risk_colonization_lag_1 = "P. aeruginosa/MRSA/ABPA", LiverDisease_lag_1 = "CF liver disease", Sex = "Sex", MRSA_lag_1 = "MRSA, anchor",
  Pancreatic_lag_1 = "Pancreatic insufficiency", HospDaysTotal_lag_1 = "Hospital days, anchor", hosp_burden_2yr_lag = "Hospital days, t−1 + t", Antibiotic_lag_1 = "Inhaled antibiotic",
  Azithro_lag_1 = "Azithromycin", NaCl_lag_1 = "Hypertonic saline", Diabetes_lag_1 = "CFRD", ABPA_lag_1 = "ABPA", MSSA_lag_1 = "MSSA", iv_both_years = "IV in both years",
  pex_burden_2yr_lag = "Exacerbations, t−1 + t", pa_both_years = "P. aeruginosa both years", pa_new_t = "New P. aeruginosa", n_pathogens_t = "Number of pathogens",
  treatment_burden_t = "Number of maintenance therapies", resp_support_t = "Oxygen or NIV", years_since_diagnosis = "Years since diagnosis",
  Afumigatus_lag_1 = "A. fumigatus", StenotrophMaltophilia_lag_1 = "S. maltophilia", HaemInf_lag_1 = "H. influenzae", PPI_lag_1 = "Proton-pump inhibitor",
  Bronchodil_lag_1 = "Bronchodilator", SteroidInhaled_lag_1 = "Inhaled steroid", pa_any_2y = "P. aeruginosa in t−1 or t", mrsa_any_2y = "MRSA in t−1 or t", abpa_any_2y = "ABPA in t−1 or t",
  cftr_n_f508del = "F508del alleles", cftr_severe = "Severe genotype (class I–III)", cftr_mild = "Mild genotype (class IV–VI)", cftr_eti_eligible = "ETI-eligible genotype",
  cftr_incomplete = "Genotype incomplete")
th_shap <- theme_classic(base_size = 9, base_family = FONT) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), axis.text.y = element_text(size = 8.5, colour = "grey10"),
        axis.line.x = element_line(colour = "grey35", linewidth = 0.4), axis.text.x = element_text(colour = "grey25"),
        plot.title = element_text(face = "bold", size = 9.5), plot.tag = element_text(face = "bold", size = 13))
shap_panel <- function(outcome_, xlab, n = 12, keyh = 32) {
  m <- read_csv(file.path(FD, paste0("shap_mean_", outcome_, ".csv")), show_col_types = FALSE)
  d <- read_csv(file.path(FD, paste0("shap_rows_", outcome_, ".csv")), show_col_types = FALSE) |> filter(!is.na(shap))
  lev <- rev(head(m$feature, n)); set.seed(1)
  d <- d |> filter(feature %in% lev) |> group_by(feature) |> mutate(dens = { k <- density(shap, n = 512); approx(k$x, k$y, shap)$y }, off = runif(n(), -1, 1) * 0.40 * dens / max(dens)) |>
    ungroup() |> mutate(yy = match(feature, lev) + off) |> arrange(!is.na(value_pct))
  ggplot(d, aes(shap, yy, colour = value_pct)) +
    geom_hline(yintercept = seq_along(lev), colour = "#cccccc", linewidth = 0.3, linetype = "22") +
    geom_vline(xintercept = 0, colour = "#999999", linewidth = 0.4) + geom_point(size = 0.55, alpha = 0.95, stroke = 0) +
    scale_colour_gradientn(colours = SHAP_PAL, limits = c(0, 1), na.value = "#777777", breaks = c(0.03, 0.97), labels = c("Low", "High"), name = "Feature value",
                           guide = guide_colourbar(theme = theme(legend.key.height = unit(keyh, "mm"), legend.key.width = unit(2.2, "mm"), legend.ticks = element_blank()))) +
    scale_y_continuous(breaks = seq_along(lev), labels = m$label[match(lev, m$feature)], expand = expansion(add = 0.6)) +
    labs(x = xlab, y = NULL) + th_shap +
    theme(legend.position = "right", legend.title = element_text(size = 7.5, angle = 90, hjust = 0.5), legend.title.position = "right",
          legend.text = element_text(size = 7.5), legend.margin = margin(0, 0, 0, -2)) }
grp_panel <- function(outcome_) { g <- read_csv(file.path(FD, paste0("shap_grouped_", outcome_, ".csv")), show_col_types = FALSE)
  ggplot(g, aes(mean_abs, reorder(group, mean_abs))) + geom_col(fill = "#ff0051", width = 0.62) +
    geom_text(aes(label = sprintf("+%.2f", mean_abs)), hjust = -0.15, size = 2.6, family = FONT, colour = "#ff0051") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.24))) + labs(x = "mean(|SHAP value|), grouped", y = NULL) + th_shap +
    theme(panel.grid.major.y = element_blank()) }
save((shap_panel("y_pex", "SHAP value (impact on log-odds)", n = 23) + labs(title = "Machine learning: IV-treated exacerbation", tag = "A")) +
       (grp_panel("y_pex") + labs(title = "Grouped contributions", tag = "B")) + plot_layout(widths = c(1.35, 1)), "FigureS3_shap_exacerbation", 8.4, 5.2)
# ---------------- Figure 3: the two exacerbation tools
p3a <- fS1 + labs(tag = "A", title = "CFRT exacerbation score: points chart") + theme(plot.tag = element_text(face = "bold", size = 13), plot.tag.position = c(0.01, 0.99),
                                                                          plot.title = element_text(face = "bold", size = 9.5, hjust = 0.045))
p3b <- (calib_main("y_pex") + labs(title = "Predicted versus observed risk", tag = "B")) / calib_hist("y_pex") + plot_layout(heights = c(3.3, 1))
p3c <- shap_panel("y_pex", "SHAP value (impact on log-odds)", n = 10, keyh = 22) + labs(title = "Machine learning: main factors", tag = "C") + theme(plot.title.position = "plot")
save(wrap_elements(full = p3a) / (wrap_elements(full = p3b) | wrap_elements(full = p3c)) + plot_layout(heights = c(1.05, 1)), "Figure3_exacerbation_tools", 7.6, 8.8)
save((shap_panel("y_fev1_pp_change", "SHAP value (points of FEV1 % predicted)") + labs(title = "Machine learning: change in FEV1 to next year", tag = "A")) +
       (grp_panel("y_fev1_pp_change") + labs(title = "Grouped contributions", tag = "B")) + plot_layout(widths = c(1.35, 1)), "FigureS4_shap_fev1", 8.4, 4.2)

# ---------------- forest helper
forest <- function(d, x = "auc", lo = "lo", hi = "hi", xlab = "AUC (95% CI)", xlim = c(0.35, 1), ref = 0.5) {
  ggplot(d, aes(.data[[x]], lab, colour = model)) + geom_vline(xintercept = ref, linetype = 3, colour = "grey50") +
    geom_errorbar(aes(xmin = .data[[lo]], xmax = .data[[hi]]), width = 0.25, linewidth = 0.45, orientation = "y", position = position_dodge(0.6)) +
    geom_point(size = 1.5, position = position_dodge(0.6)) + scale_colour_manual(values = COL) + coord_cartesian(xlim = xlim) + labs(x = xlab, y = NULL) + th + theme(legend.position = "bottom") }

# ---------------- Figure S5: leave-one-centre-out (AUC and O/E)
lo <- read_csv(file.path(FD, "loco.csv"), show_col_types = FALSE) |> filter(model %in% c("CFRT exacerbation score", "Machine learning", "IV-antibiotic days alone"))
mk <- function(o, metric, ttl) { d <- lo |> filter(outcome == o) |> mutate(lab = ifelse(group == "Random-effects pooled", group, sprintf("%s (%d/%d)", group, as.integer(events), as.integer(n))))
  inst <- d |> filter(group != "Random-effects pooled") |> distinct(lab, group) |> mutate(o = ifelse(grepl("^Other", group), "ZZ", group)) |> arrange(o)
  d$lab <- factor(d$lab, levels = rev(c("Random-effects pooled", inst$lab))); pooled <- d |> filter(group == "Random-effects pooled")
  if (metric == "auc") { p <- forest(d |> filter(group != "Random-effects pooled")) +
      geom_errorbar(data = pooled, aes(xmin = pi_lo, xmax = pi_hi), width = 0, linewidth = 2.2, alpha = 0.25, orientation = "y", position = position_dodge(0.6)) +
      geom_errorbar(data = pooled, aes(xmin = lo, xmax = hi), width = 0.25, linewidth = 0.6, orientation = "y", position = position_dodge(0.6)) +
      geom_point(data = pooled, shape = 18, size = 3, position = position_dodge(0.6)) }
  else { p <- ggplot(d |> filter(group != "Random-effects pooled"), aes(oe, lab, colour = model)) + geom_vline(xintercept = 1, linetype = 3) +
      geom_point(size = 1.5, position = position_dodge(0.6)) + geom_errorbar(data = pooled, aes(xmin = oe_pi_lo, xmax = oe_pi_hi), width = 0, linewidth = 2.2, alpha = 0.25, orientation = "y", position = position_dodge(0.6)) +
      geom_errorbar(data = pooled, aes(xmin = oe_lo, xmax = oe_hi), width = 0.25, linewidth = 0.6, orientation = "y", position = position_dodge(0.6)) +
      geom_point(data = pooled, shape = 18, size = 3, position = position_dodge(0.6)) + scale_colour_manual(values = COL) + scale_x_log10() +
      labs(x = "Observed/expected (log scale)", y = NULL) + th + theme(legend.position = "bottom") }
  p + ggtitle(ttl) }
fs5 <- (mk("y_pex", "auc", "Discrimination") + theme(legend.position = "bottom")) + (mk("y_pex", "oe", "Calibration-in-the-large") + theme(legend.position = "none"))
save(fs5 + plot_annotation(tag_levels = "A", caption = "Held-out centres (events/prediction-years). Diamond: random-effects pooled estimate; shaded bar: 95% prediction interval."), "FigureS5_leave_centre_out", 8.0, 4.4)

# ---------------- Figure S6: subgroups (AUC and O/E)
sg <- read_csv(file.path(FD, "subgroups.csv"), show_col_types = FALSE) |> filter(outcome == "y_pex", !is.na(auc)) |>
  mutate(lab = sprintf("%s (%d/%d)", group, events, n)); sg$lab <- factor(sg$lab, levels = rev(unique(sg$lab)))
fs6 <- (forest(sg, "auc", "auc_lo", "auc_hi") + ggtitle("Discrimination")) + (forest(sg, "oe", "oe_lo", "oe_hi", "Observed/expected", c(0.3, 2.4), 1) + ggtitle("Calibration-in-the-large") +
  theme(axis.text.y = element_blank())) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
save(fs6 + plot_annotation(tag_levels = "A"), "FigureS6_subgroups", 8.0, 6.4)

# ---------------- Figure S7: calendar year (registry, modulator-untreated records, age >5 y)
cy <- read_csv(file.path(FD, "calendar_year.csv"), show_col_types = FALSE) |> select(year, iv_any, spirometry_recorded, pa_any) |>
  pivot_longer(-year) |> mutate(name = recode(name, iv_any = "Any IV antibiotics", spirometry_recorded = "Spirometry recorded", pa_any = "P. aeruginosa isolated"))
fs7 <- ggplot(cy, aes(year, value, colour = name)) + annotate("rect", xmin = 2019.6, xmax = 2021.4, ymin = -Inf, ymax = Inf, fill = "grey93") +
  geom_line() + geom_point(size = 1.3) + scale_y_continuous(labels = percent) + scale_colour_manual(values = c("#0072B2", "#D55E00", "#009E73")) +
  labs(x = "Registry year", y = "Proportion of modulator-untreated records", caption = "Shaded: COVID-19 pandemic years 2020–2021.") + th + theme(legend.position = "bottom")
save(fs7, "FigureS7_calendar_year", 5.2, 3.6)
cat("R figures written\n")
