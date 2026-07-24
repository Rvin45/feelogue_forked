
LOAD_CHART_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "description": "Charts that match the user's request, ordered by relevance (best match first). Empty if nothing matches.",
            "items": {
                "type": "object",
                "properties": {
                    "chart_id": {
                        "type": "integer",
                        "description": "The unique ID of the chart"
                    },
                    "chart_name": {
                        "type": "string",
                        "description": "The display name of the chart"
                    }
                },
                "required": ["chart_id", "chart_name"],
                "additionalProperties": False
            }
        }
    },
    "required": ["matches"],
    "additionalProperties": False
}


# Formatted output for ChatGPT response
INTENT_SCHEMA={
  "type": "object",
  "properties": {
    "intents": {
      "type": "array",
      "description": "List of detected user intents, ordered by priority (most important first)",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "load_chart",
              "image_analysis",
              "chart_overview",
              "touch_interaction",
              "data_analysis",
              "trend",
              "operations",
              "general_question"
            ],
            "description": "The classified intent type"
          },
          "query": {
            "type": "string",
            "description": "A fully self-contained query specific to this intent"
          }
        },
        "required": ["type", "query"],
        "additionalProperties": False
      },
      "minItems": 1
    },
    "has_deictic": {
      "type": "boolean",
      "description": "True if the query includes explicit deictic references like 'this', 'that', or touched elements"
    }
  },
  "required": ["intents", "has_deictic"],
  "additionalProperties": False
}

# Structured extraction for chart view operations
# (zoom / zoom_in / zoom_out / pan / filter / remove_filter / reset)
OPERATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {
            "type": ["string", "null"],
            "enum": ["zoom", "zoom_in", "zoom_out", "pan",
                     "filter", "remove_filter", "reset", None],
            "description": (
                "The view operation, or null if unsupported/unclassifiable. "
                "zoom = scale toward a named destination; REQUIRES a target or an "
                "explicit factor (both null is invalid — that utterance is zoom_in). "
                "zoom_in / zoom_out = bare directional step zooms ('zoom in', "
                "'zoom out a bit'); ALWAYS target null and factor null. "
                "pan = change position, scale unchanged. "
                "filter = HIDE the target (filter always means hide; no keep); "
                "REQUIRES a non-null target. "
                "remove_filter = re-show previously hidden data; target null means "
                "clear ALL filters, a target means re-show that point/range/series. "
                "reset = default full view; always target null, factor null. "
                "null = data questions, layer switches, undo, or anything too "
                "ambiguous — always with clarification_needed true."
            )
        },
        "target": {
            "type": ["array", "null"],
            "description": (
                "Resolved targets, or null when the operation takes none. Each "
                "element is {axis, value}: "
                "axis 'x' or 'y' for data-space coordinates (ISO date string for "
                "temporal axes, number for numeric axes); "
                "axis 'series' for a series name from the data context (filter and "
                "remove_filter only); "
                "axis null for exactly one of the 8 compass tokens: north, south, "
                "east, west, north_east, north_west, south_east, south_west "
                "(normalise synonyms: left→west, up→north, 'top right'→north_east; "
                "all 8 valid for pan; only the 4 intercardinals valid for zoom, as "
                "quadrant regions; token elements are never combined with values). "
                "Composition: one element = a point on that axis, a direction/"
                "quadrant, or a whole series; two elements on the SAME axis = a "
                "range [start, end], start < end; one x plus one y element = a "
                "single 2D point; a series element plus point/range elements = a "
                "scoped filter, hiding only points matching ALL elements "
                "(conjunction; element order carries no meaning). "
                "Coordinates need not be existing data points — empty chart regions "
                "are legal targets; validate against extents, never snap to the "
                "nearest point. Only ordinals, features, deixis, and relative "
                "references must resolve to existing points (via touch or the query "
                "tool). Never emit unresolved phrases ('the max', 'second point'), "
                "row indices, or screen coordinates."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": ["string", "null"],
                        "enum": ["x", "y", "series", None]
                    },
                    "value": {"type": ["string", "number"]}
                },
                "required": ["axis", "value"],
                "additionalProperties": False
            }
        },
        "factor": {
            "type": ["number", "null"],
            "description": (
                "ONLY the magnitude the user explicitly spoke, as an absolute zoom "
                "percent (e.g. 175; '2x'/'double' → 200). Valid for zoom only. Null "
                "in every other case — never fill in a default. Range targets never "
                "take a factor (the fit is derived); pan never takes a percent."
            )
        },
        "clarification_needed": {
            "type": "boolean",
            "description": (
                "True if too ambiguous to confidently produce operation/target/"
                "factor, or the request is unsupported. Best-supported partial "
                "reading stays filled; null only what cannot be filled."
            )
        },
        "message": {
            "type": "string",
            "description": (
                "Spoken via TTS: one short sentence, no symbols, 'percent' in "
                "words. If clarification_needed: one question, answerable in a "
                "word or two, options included. Otherwise: confirm the RESOLVED "
                "interpretation (say what 'the second data point' resolved to; "
                "echo the user's own direction word, not the token)."
            )
        }
    },
    "required": ["operation", "target", "factor",
                 "clarification_needed", "message"],
    "additionalProperties": False
}