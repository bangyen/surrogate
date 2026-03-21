use crate::ml::model::PhaseEnsemble;
use ndarray::Array1;
use std::collections::HashMap;

pub struct SurrogateExplainer {
    pub model: PhaseEnsemble,
    pub feature_templates: HashMap<String, String>,
}

impl SurrogateExplainer {
    pub fn new(model: PhaseEnsemble) -> Self {
        let mut feature_templates = HashMap::new();
        let templates = [
            (
                "material_diff",
                "Gains a **material advantage** ({:+.0} cp)",
            ),
            (
                "material_us",
                "Increases **total material value** ({:+.0} cp)",
            ),
            (
                "material_them",
                "Reduces opponent's **total material** ({:+.0} cp)",
            ),
            (
                "mobility_us",
                "Increases **piece activity** and **mobility** ({:+.0} cp)",
            ),
            (
                "mobility_them",
                "Restricts opponent's **piece activity** ({:+.0} cp)",
            ),
            (
                "king_ring_pressure_us",
                "Increases **attacking pressure** near the opponent's king ({:+.0} cp)",
            ),
            (
                "king_ring_pressure_them",
                "Reduces **attacking pressure** on our own king ({:+.0} cp)",
            ),
            (
                "king_safety_us",
                "Improves the **defensive safety** of our king ({:+.0} cp)",
            ),
            (
                "king_safety_them",
                "Exposes or weakens the opponent's **king safety** ({:+.0} cp)",
            ),
            (
                "king_pawn_shield_us",
                "Maintains a solid **pawn shield** for our king ({:+.0} cp)",
            ),
            (
                "king_pawn_shield_them",
                "Breaks through the opponent's **king pawn shield** ({:+.0} cp)",
            ),
            (
                "king_tropism_us",
                "Positions more pieces **closer to the enemy king** ({:+.0} cp)",
            ),
            (
                "king_tropism_them",
                "Keeps opponent pieces **away from our king** ({:+.0} cp)",
            ),
            (
                "piece_activity_us",
                "Maximizes the **coordination and activity** of our pieces ({:+.0} cp)",
            ),
            (
                "piece_activity_them",
                "Disrupts the **coordination** of opponent's pieces ({:+.0} cp)",
            ),
            (
                "center_control_us",
                "Improves control over the **critical central squares** ({:+.0} cp)",
            ),
            (
                "center_control_them",
                "Challenges and reduces opponent's **central control** ({:+.0} cp)",
            ),
            (
                "space_us",
                "Gains a **territorial space advantage** ({:+.0} cp)",
            ),
            (
                "space_them",
                "Cramps the opponent's position and reduces their **space** ({:+.0} cp)",
            ),
            (
                "batteries_us",
                "Forms a powerful **battery arrangement** ({:+.0} cp)",
            ),
            (
                "batteries_them",
                "Dismantles or blocks opponent's **batteries** ({:+.0} cp)",
            ),
            (
                "outposts_us",
                "Establishes a strong **knight outpost** ({:+.0} cp)",
            ),
            (
                "outposts_them",
                "Challenges or eliminates an opponent's **outpost** ({:+.0} cp)",
            ),
            (
                "bishop_pair_us",
                "Maintains the **bishop pair advantage** ({:+.0} cp)",
            ),
            (
                "bishop_pair_them",
                "Eliminates the opponent's **bishop pair** ({:+.0} cp)",
            ),
            (
                "passed_us",
                "Creates a dangerous **passed pawn** ({:+.0} cp)",
            ),
            (
                "passed_them",
                "Successfully blocks or stops an opponent's **passed pawn** ({:+.0} cp)",
            ),
            (
                "isolated_pawns_us",
                "Avoids creating **pawn weaknesses** ({:+.0} cp)",
            ),
            (
                "isolated_pawns_them",
                "Forces a **pawn weakness** (isolated pawn) for the opponent ({:+.0} cp)",
            ),
            (
                "doubled_pawns_us",
                "Fixes or avoids **doubled pawn** structural weaknesses ({:+.0} cp)",
            ),
            (
                "doubled_pawns_them",
                "Induces **doubled pawn** weaknesses for the opponent ({:+.0} cp)",
            ),
            (
                "backward_pawns_us",
                "Solidifies the pawn structure by fixing a **weakness** ({:+.0} cp)",
            ),
            (
                "backward_pawns_them",
                "Induces a **backward pawn weakness** in the opponent's camp ({:+.0} cp)",
            ),
            (
                "pawn_chain_us",
                "Creates a solid and supportive **pawn chain** ({:+.0} cp)",
            ),
            (
                "pawn_chain_them",
                "Breaks up the opponent's **pawn chain** ({:+.0} cp)",
            ),
            (
                "safe_mobility_us",
                "Safely **activates pieces** to better squares ({:+.0} cp)",
            ),
            (
                "safe_mobility_them",
                "Restricts the **safe movement** of opponent's pieces ({:+.0} cp)",
            ),
            (
                "rook_open_file_us",
                "Positions a rook effectively on an **open file** ({:+.0} cp)",
            ),
            (
                "rook_open_file_them",
                "Denies the opponent control of **open files** ({:+.0} cp)",
            ),
            (
                "rook_on_7th_us",
                "Positions a rook dangerously on the **7th rank** ({:+.0} cp)",
            ),
            (
                "connected_rooks_us",
                "**Connects rooks** for mutual support and power ({:+.0} cp)",
            ),
            (
                "pinned_us",
                "Successfully escapes an **annoying pin** ({:+.0} cp)",
            ),
            (
                "pinned_them",
                "**Pins** an opponent's piece to create tactical opportunities ({:+.0} cp)",
            ),
            (
                "hanging_us",
                "Defends or moves a **hanging piece** ({:+.0} cp)",
            ),
            (
                "hanging_them",
                "Exploits or creates a **hanging piece** for the opponent ({:+.0} cp)",
            ),
            (
                "threats_us",
                "Creates immediate **tactical threats** ({:+.0} cp)",
            ),
            (
                "threats_them",
                "Neutralizes or parries opponent's **threats** ({:+.0} cp)",
            ),
            (
                "phase",
                "Strategic move appropriate for the current game phase ({:+.0} cp)",
            ),
        ];

        for (k, v) in templates {
            feature_templates.insert(k.to_string(), v.to_string());
        }

        SurrogateExplainer {
            model,
            feature_templates,
        }
    }

    pub fn get_feature_label(&self, name: &str) -> String {
        self.feature_templates
            .get(name)
            .map(|t| {
                t.replace(" ({:+.0} cp)", "")
                    .replace(" ({:+.1} cp)", "")
                    .replace("**", "")
            })
            .unwrap_or_else(|| {
                name.replace('_', " ")
                    .split_whitespace()
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
    }

    pub fn explain_move(
        &self,
        features_after: &std::collections::BTreeMap<String, f32>,
        top_k: usize,
        min_cp: f64,
    ) -> Vec<(String, f64, String)> {
        let mut reasons = Vec::new();

        let mut delta_vec = Array1::zeros(self.model.feature_names.len());
        for (i, name) in self.model.feature_names.iter().enumerate() {
            delta_vec[i] = *features_after.get(name).unwrap_or(&0.0) as f64;
        }

        let delta_scaled = if let Some(scaler) = &self.model.scaler {
            let mut s = delta_vec.clone();
            s -= &scaler.mean;
            s /= &scaler.scale;
            s
        } else {
            delta_vec.clone()
        };

        let contributions = self.model.get_contributions(&delta_scaled.view());

        let mut significant = Vec::new();
        for (i, &contrib) in contributions.iter().enumerate() {
            let cp_value: f64 = contrib;
            if cp_value.abs() >= min_cp {
                significant.push((self.model.feature_names[i].clone(), cp_value));
            }
        }

        significant.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for (name, cp_value) in significant.into_iter().take(top_k) {
            let template = self
                .feature_templates
                .get(&name)
                .cloned()
                .unwrap_or_else(|| format!("{} ({:+.1} cp)", self.get_feature_label(&name), cp_value));
            let explanation = template.replace("{:+.0}", &format!("{:+.0}", cp_value));
            reasons.push((name, cp_value, explanation));
        }

        reasons
    }

    pub fn get_formatted_features(
        &self,
        features: &std::collections::BTreeMap<String, f32>,
    ) -> Vec<(String, String, f32)> {
        let mut formatted = Vec::new();
        for (name, &val) in features {
            formatted.push((name.clone(), self.get_feature_label(name), val));
        }
        formatted.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        formatted
    }
}
