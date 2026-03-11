#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define LINE_BUFFER_SIZE 8192
#define FIELD_COUNT 5

/*
 * Too slow for Python, switching to C.
 */

static const char *FIELD_NAMES[FIELD_COUNT] = {
    "Discharge",
    "Charge",
    "Production",
    "Consumption",
    "State of Charge"
};

typedef struct {
    char *timestamp;
    double values[FIELD_COUNT];
    bool valid[FIELD_COUNT];
} Row;

typedef struct {
    Row *rows;
    size_t count;
    size_t capacity;
} RowBuffer;

typedef struct {
    bool has_result;
    int shift;
    size_t compared;
    double mae;
    double within_tolerance_pct;
    double r_squared;
} ShiftResult;

typedef struct {
    int coarse_step;
    int sample_stride;
    int refine_window;
} SearchConfig;

static char *duplicate_string(const char *source);
static bool row_buffer_push(RowBuffer *buffer, const Row *row);

static int64_t days_from_civil(int year, int month, int day) {
    int adjusted_year = year - (month <= 2);
    int era = (adjusted_year >= 0 ? adjusted_year : adjusted_year - 399) / 400;
    unsigned yoe = (unsigned)(adjusted_year - era * 400);
    unsigned adjusted_month = (unsigned)(month + (month > 2 ? -3 : 9));
    unsigned doy = (153U * adjusted_month + 2U) / 5U + (unsigned)day - 1U;
    unsigned doe = yoe * 365U + yoe / 4U - yoe / 100U + doy;
    return era * 146097 + (int64_t)doe - 719468;
}

static void civil_from_days(int64_t z, int *year, int *month, int *day) {
    int era;
    unsigned doe;
    unsigned yoe;
    int y;
    unsigned doy;
    unsigned mp;

    z += 719468;
    era = (int)(z >= 0 ? z : z - 146096) / 146097;
    doe = (unsigned)(z - (int64_t)era * 146097);
    yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = (int)yoe + era * 400;
    doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    mp = (5 * doy + 2) / 153;
    *day = (int)(doy - (153 * mp + 2) / 5 + 1);
    *month = (int)(mp + (mp < 10 ? 3 : -9));
    *year = y + (*month <= 2);
}

static bool parse_timestamp_to_minute(const char *timestamp, int64_t *out_minute) {
    int year;
    int month;
    int day;
    int hour;
    int minute;
    int second;
    int parsed;
    int64_t days;

    if (timestamp == NULL || out_minute == NULL) {
        return false;
    }

    parsed = sscanf(timestamp, "%d-%d-%d %d:%d:%d", &year, &month, &day, &hour, &minute, &second);
    if (parsed != 6) {
        return false;
    }
    if (month < 1 || month > 12 || day < 1 || day > 31 || hour < 0 || hour > 23 ||
        minute < 0 || minute > 59 || second < 0 || second > 59) {
        return false;
    }

    days = days_from_civil(year, month, day);
    *out_minute = days * 1440 + (int64_t)hour * 60 + minute;
    return true;
}

static char *format_minute_timestamp(int64_t absolute_minute) {
    int64_t days = absolute_minute / 1440;
    int64_t day_minute = absolute_minute % 1440;
    int year;
    int month;
    int day;
    int hour;
    int minute;
    char *text = (char *)malloc(32);

    if (text == NULL) {
        return NULL;
    }
    if (day_minute < 0) {
        day_minute += 1440;
        days -= 1;
    }

    civil_from_days(days, &year, &month, &day);
    hour = (int)(day_minute / 60);
    minute = (int)(day_minute % 60);
    snprintf(text, 32, "%04d-%02d-%02d %02d:%02d:00", year, month, day, hour, minute);
    return text;
}

static bool push_row_duplicate(RowBuffer *buffer, const Row *source_row) {
    Row row_copy;

    if (buffer == NULL || source_row == NULL) {
        return false;
    }

    row_copy = *source_row;
    row_copy.timestamp = duplicate_string(source_row->timestamp);
    if (row_copy.timestamp == NULL) {
        return false;
    }

    if (!row_buffer_push(buffer, &row_copy)) {
        free(row_copy.timestamp);
        return false;
    }
    return true;
}

static bool push_missing_row(RowBuffer *buffer, int64_t absolute_minute) {
    Row missing_row;
    int field_index;

    memset(&missing_row, 0, sizeof(missing_row));
    missing_row.timestamp = format_minute_timestamp(absolute_minute);
    if (missing_row.timestamp == NULL) {
        return false;
    }
    for (field_index = 0; field_index < FIELD_COUNT; ++field_index) {
        missing_row.valid[field_index] = false;
        missing_row.values[field_index] = 0.0;
    }

    if (!row_buffer_push(buffer, &missing_row)) {
        free(missing_row.timestamp);
        return false;
    }
    return true;
}

static bool expand_to_minute_grid(const RowBuffer *source, RowBuffer *expanded) {
    size_t index;

    if (source == NULL || expanded == NULL) {
        return false;
    }
    if (source->count == 0U) {
        return true;
    }

    for (index = 0; index < source->count; ++index) {
        int64_t current_minute;
        if (!push_row_duplicate(expanded, &source->rows[index])) {
            return false;
        }

        if (index + 1U >= source->count) {
            continue;
        }

        if (parse_timestamp_to_minute(source->rows[index].timestamp, &current_minute)) {
            int64_t next_minute;
            if (parse_timestamp_to_minute(source->rows[index + 1U].timestamp, &next_minute) &&
                next_minute > current_minute + 1) {
                int64_t missing_minute;
                for (missing_minute = current_minute + 1; missing_minute < next_minute;
                     ++missing_minute) {
                    if (!push_missing_row(expanded, missing_minute)) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

static void apply_next_after_invalid_mask(RowBuffer *buffer) {
    size_t row_index;
    int field_index;

    if (buffer == NULL || buffer->count < 2U) {
        return;
    }

    for (field_index = 0; field_index < FIELD_COUNT; ++field_index) {
        bool *previous_invalid = (bool *)calloc(buffer->count, sizeof(bool));
        if (previous_invalid == NULL) {
            return;
        }

        for (row_index = 1U; row_index < buffer->count; ++row_index) {
            previous_invalid[row_index] = !buffer->rows[row_index - 1U].valid[field_index];
        }

        for (row_index = 0U; row_index < buffer->count; ++row_index) {
            if (previous_invalid[row_index]) {
                buffer->rows[row_index].valid[field_index] = false;
            }
        }
        free(previous_invalid);
    }
}

static void forward_fill_columns(RowBuffer *buffer) {
    int field_index;
    size_t row_index;

    if (buffer == NULL) {
        return;
    }

    for (field_index = 0; field_index < FIELD_COUNT; ++field_index) {
        bool has_previous = false;
        double previous_value = 0.0;

        for (row_index = 0U; row_index < buffer->count; ++row_index) {
            if (buffer->rows[row_index].valid[field_index]) {
                previous_value = buffer->rows[row_index].values[field_index];
                has_previous = true;
                continue;
            }
            if (has_previous) {
                buffer->rows[row_index].values[field_index] = previous_value;
                buffer->rows[row_index].valid[field_index] = true;
            }
        }
    }
}

static bool preprocess_like_energy_plots(const RowBuffer *source, RowBuffer *processed) {
    if (!expand_to_minute_grid(source, processed)) {
        return false;
    }
    apply_next_after_invalid_mask(processed);
    forward_fill_columns(processed);
    return true;
}

static void free_row_buffer(RowBuffer *buffer) {
    size_t index;
    if (buffer == NULL || buffer->rows == NULL) {
        return;
    }

    for (index = 0; index < buffer->count; ++index) {
        free(buffer->rows[index].timestamp);
    }

    free(buffer->rows);
    buffer->rows = NULL;
    buffer->count = 0;
    buffer->capacity = 0;
}

static char *duplicate_string(const char *source) {
    size_t length;
    char *copy;

    if (source == NULL) {
        return NULL;
    }

    length = strlen(source);
    copy = (char *)malloc(length + 1);
    if (copy == NULL) {
        return NULL;
    }

    memcpy(copy, source, length + 1);
    return copy;
}

static void trim_trailing(char *text) {
    size_t length;

    if (text == NULL) {
        return;
    }

    length = strlen(text);
    while (length > 0) {
        char current = text[length - 1];
        if (current != '\n' && current != '\r' && current != ' ' && current != '\t') {
            break;
        }
        text[length - 1] = '\0';
        --length;
    }
}

static char *trim_leading(char *text) {
    if (text == NULL) {
        return NULL;
    }

    while (*text == ' ' || *text == '\t') {
        ++text;
    }
    return text;
}

static bool parse_double_field(const char *text, double *out_value) {
    char *endptr = NULL;
    double parsed;

    if (text == NULL || out_value == NULL) {
        return false;
    }

    errno = 0;
    parsed = strtod(text, &endptr);
    if (text == endptr || errno != 0) {
        return false;
    }

    while (*endptr == ' ' || *endptr == '\t') {
        ++endptr;
    }

    if (*endptr != '\0') {
        return false;
    }

    *out_value = parsed;
    return true;
}

static bool row_buffer_push(RowBuffer *buffer, const Row *row) {
    Row *new_rows;
    size_t new_capacity;

    if (buffer == NULL || row == NULL) {
        return false;
    }

    if (buffer->count == buffer->capacity) {
        new_capacity = (buffer->capacity == 0U) ? 4096U : (buffer->capacity * 2U);
        new_rows = (Row *)realloc(buffer->rows, new_capacity * sizeof(Row));
        if (new_rows == NULL) {
            return false;
        }
        buffer->rows = new_rows;
        buffer->capacity = new_capacity;
    }

    buffer->rows[buffer->count] = *row;
    buffer->count += 1U;
    return true;
}

static bool get_token_at_index(char *line, int target_index, char **out_token) {
    int index = 0;
    char *token;

    token = strtok(line, ",");
    while (token != NULL) {
        if (index == target_index) {
            *out_token = token;
            return true;
        }
        token = strtok(NULL, ",");
        ++index;
    }

    return false;
}

static bool parse_line_to_row(char *line, const int *column_map, Row *row) {
    char line_copy[LINE_BUFFER_SIZE];
    char *token;
    int field_index;

    if (line == NULL || column_map == NULL || row == NULL) {
        return false;
    }

    trim_trailing(line);
    if (*line == '\0') {
        return false;
    }

    strncpy(line_copy, line, sizeof(line_copy) - 1U);
    line_copy[sizeof(line_copy) - 1U] = '\0';

    if (!get_token_at_index(line_copy, 0, &token)) {
        return false;
    }

    token = trim_leading(token);
    row->timestamp = duplicate_string(token);
    if (row->timestamp == NULL) {
        return false;
    }

    for (field_index = 0; field_index < FIELD_COUNT; ++field_index) {
        char value_copy[LINE_BUFFER_SIZE];
        char *value_token;
        bool ok;
        int target_col = column_map[field_index];

        row->valid[field_index] = false;
        row->values[field_index] = 0.0;

        strncpy(value_copy, line, sizeof(value_copy) - 1U);
        value_copy[sizeof(value_copy) - 1U] = '\0';

        if (!get_token_at_index(value_copy, target_col, &value_token)) {
            continue;
        }

        value_token = trim_leading(value_token);
        ok = parse_double_field(value_token, &row->values[field_index]);
        row->valid[field_index] = ok;
    }

    return true;
}

static bool load_file_rows(const char *path, const int *column_map, RowBuffer *out_buffer) {
    FILE *file;
    char line[LINE_BUFFER_SIZE];

    if (path == NULL || column_map == NULL || out_buffer == NULL) {
        return false;
    }

    file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open %s\n", path);
        return false;
    }

    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        fprintf(stderr, "File is empty: %s\n", path);
        return false;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        Row row;
        bool parsed = parse_line_to_row(line, column_map, &row);
        if (!parsed) {
            continue;
        }
        if (!row_buffer_push(out_buffer, &row)) {
            free(row.timestamp);
            fclose(file);
            fprintf(stderr, "Out of memory while loading %s\n", path);
            return false;
        }
    }

    fclose(file);
    return true;
}

static ShiftResult evaluate_shift(
    const RowBuffer *w_rows,
    const RowBuffer *wh_rows,
    int field_index,
    int shift,
    bool convert_w_to_wh,
    double tolerance
) {
    ShiftResult result;
    size_t i;
    size_t w_start;
    size_t wh_start;
    size_t length;
    double abs_sum = 0.0;
    size_t compared = 0U;
    size_t within = 0U;
    double observed_sum = 0.0;
    double observed_sq_sum = 0.0;
    double sse_sum = 0.0;

    result.has_result = false;
    result.shift = shift;
    result.compared = 0U;
    result.mae = 0.0;
    result.within_tolerance_pct = 0.0;
    result.r_squared = NAN;

    if (w_rows == NULL || wh_rows == NULL || field_index < 0 || field_index >= FIELD_COUNT) {
        return result;
    }

    if (shift >= 0) {
        w_start = 0U;
        wh_start = (size_t)shift;
        if (wh_start >= wh_rows->count) {
            return result;
        }
        length = w_rows->count;
        if (wh_rows->count - wh_start < length) {
            length = wh_rows->count - wh_start;
        }
    } else {
        w_start = (size_t)(-shift);
        wh_start = 0U;
        if (w_start >= w_rows->count) {
            return result;
        }
        length = wh_rows->count;
        if (w_rows->count - w_start < length) {
            length = w_rows->count - w_start;
        }
    }

    for (i = 0; i < length; ++i) {
        size_t wi = w_start + i;
        size_t hi = wh_start + i;
        double expected;
        double observed;
        double error;

        if (!w_rows->rows[wi].valid[field_index] || !wh_rows->rows[hi].valid[field_index]) {
            continue;
        }

        expected = w_rows->rows[wi].values[field_index];
        if (convert_w_to_wh && field_index != 4) {
            expected /= 60.0;
        }

        observed = wh_rows->rows[hi].values[field_index];
        error = fabs(observed - expected);

        abs_sum += error;
        sse_sum += error * error;
        observed_sum += observed;
        observed_sq_sum += observed * observed;
        compared += 1U;
        if (error <= tolerance) {
            within += 1U;
        }
    }

    if (compared == 0U) {
        return result;
    }

    result.has_result = true;
    result.compared = compared;
    result.mae = abs_sum / (double)compared;
    result.within_tolerance_pct = (100.0 * (double)within) / (double)compared;
    if (compared > 1U) {
        double observed_mean = observed_sum / (double)compared;
        double sst = observed_sq_sum - (double)compared * observed_mean * observed_mean;
        if (sst > 1e-12) {
            result.r_squared = 1.0 - (sse_sum / sst);
        }
    }
    return result;
}

static ShiftResult evaluate_shift_sampled(
    const RowBuffer *w_rows,
    const RowBuffer *wh_rows,
    int field_index,
    int shift,
    bool convert_w_to_wh,
    double tolerance,
    int sample_stride
) {
    ShiftResult result;
    size_t i;
    size_t w_start;
    size_t wh_start;
    size_t length;
    double abs_sum = 0.0;
    size_t compared = 0U;
    size_t within = 0U;
    double observed_sum = 0.0;
    double observed_sq_sum = 0.0;
    double sse_sum = 0.0;

    result.has_result = false;
    result.shift = shift;
    result.compared = 0U;
    result.mae = 0.0;
    result.within_tolerance_pct = 0.0;
    result.r_squared = NAN;

    if (w_rows == NULL || wh_rows == NULL || field_index < 0 || field_index >= FIELD_COUNT) {
        return result;
    }

    if (sample_stride <= 0) {
        sample_stride = 1;
    }

    if (shift >= 0) {
        w_start = 0U;
        wh_start = (size_t)shift;
        if (wh_start >= wh_rows->count) {
            return result;
        }
        length = w_rows->count;
        if (wh_rows->count - wh_start < length) {
            length = wh_rows->count - wh_start;
        }
    } else {
        w_start = (size_t)(-shift);
        wh_start = 0U;
        if (w_start >= w_rows->count) {
            return result;
        }
        length = wh_rows->count;
        if (w_rows->count - w_start < length) {
            length = w_rows->count - w_start;
        }
    }

    for (i = 0; i < length; i += (size_t)sample_stride) {
        size_t wi = w_start + i;
        size_t hi = wh_start + i;
        double expected;
        double observed;
        double error;

        if (!w_rows->rows[wi].valid[field_index] || !wh_rows->rows[hi].valid[field_index]) {
            continue;
        }

        expected = w_rows->rows[wi].values[field_index];
        if (convert_w_to_wh && field_index != 4) {
            expected /= 60.0;
        }

        observed = wh_rows->rows[hi].values[field_index];
        error = fabs(observed - expected);

        abs_sum += error;
        sse_sum += error * error;
        observed_sum += observed;
        observed_sq_sum += observed * observed;
        compared += 1U;
        if (error <= tolerance) {
            within += 1U;
        }
    }

    if (compared == 0U) {
        return result;
    }

    result.has_result = true;
    result.compared = compared;
    result.mae = abs_sum / (double)compared;
    result.within_tolerance_pct = (100.0 * (double)within) / (double)compared;
    if (compared > 1U) {
        double observed_mean = observed_sum / (double)compared;
        double sst = observed_sq_sum - (double)compared * observed_mean * observed_mean;
        if (sst > 1e-12) {
            result.r_squared = 1.0 - (sse_sum / sst);
        }
    }
    return result;
}

static ShiftResult find_best_shift(
    const RowBuffer *w_rows,
    const RowBuffer *wh_rows,
    int field_index,
    int min_shift,
    int max_shift,
    double tolerance
) {
    int shift;
    int span;
    int coarse_step;
    int sample_stride;
    int refine_window;
    int coarse_best_shift;
    SearchConfig config = {15, 30, 45};
    ShiftResult best;

    best.has_result = false;
    best.shift = 0;
    best.compared = 0U;
    best.mae = 0.0;
    best.within_tolerance_pct = 0.0;
    best.r_squared = NAN;

    span = max_shift - min_shift;
    if (span <= 240) {
        for (shift = min_shift; shift <= max_shift; ++shift) {
            ShiftResult current = evaluate_shift(
                w_rows,
                wh_rows,
                field_index,
                shift,
                true,
                tolerance
            );

            if (!current.has_result) {
                continue;
            }

            if (!best.has_result || current.mae < best.mae ||
                (fabs(current.mae - best.mae) < 1e-12 &&
                 current.within_tolerance_pct > best.within_tolerance_pct)) {
                best = current;
            }
        }
        return best;
    }

    coarse_step = config.coarse_step;
    sample_stride = config.sample_stride;
    refine_window = config.refine_window;
    coarse_best_shift = min_shift;

    for (shift = min_shift; shift <= max_shift; shift += coarse_step) {
        ShiftResult current = evaluate_shift_sampled(
            w_rows,
            wh_rows,
            field_index,
            shift,
            true,
            tolerance,
            sample_stride
        );
        if (!current.has_result) {
            continue;
        }
        if (!best.has_result || current.mae < best.mae ||
            (fabs(current.mae - best.mae) < 1e-12 &&
             current.within_tolerance_pct > best.within_tolerance_pct)) {
            best = current;
            coarse_best_shift = shift;
        }
    }

    if (!best.has_result) {
        return best;
    }

    best.has_result = false;
    for (shift = coarse_best_shift - refine_window; shift <= coarse_best_shift + refine_window;
         ++shift) {
        ShiftResult current;
        if (shift < min_shift || shift > max_shift) {
            continue;
        }
        current = evaluate_shift_sampled(
            w_rows,
            wh_rows,
            field_index,
            shift,
            true,
            tolerance,
            sample_stride
        );
        if (!current.has_result) {
            continue;
        }
        if (!best.has_result || current.mae < best.mae ||
            (fabs(current.mae - best.mae) < 1e-12 &&
             current.within_tolerance_pct > best.within_tolerance_pct)) {
            best = current;
        }
    }

    coarse_best_shift = best.shift;
    best.has_result = false;
    for (shift = coarse_best_shift - 2; shift <= coarse_best_shift + 2; ++shift) {
        ShiftResult current;
        if (shift < min_shift || shift > max_shift) {
            continue;
        }
        current = evaluate_shift(
            w_rows,
            wh_rows,
            field_index,
            shift,
            true,
            tolerance
        );
        if (!current.has_result) {
            continue;
        }
        if (!best.has_result || current.mae < best.mae ||
            (fabs(current.mae - best.mae) < 1e-12 &&
             current.within_tolerance_pct > best.within_tolerance_pct)) {
            best = current;
        }
    }

    return best;
}

static double timestamp_match_pct_for_shift(
    const RowBuffer *w_rows,
    const RowBuffer *wh_rows,
    int shift
) {
    size_t i;
    size_t w_start;
    size_t wh_start;
    size_t length;
    size_t matched = 0U;

    if (w_rows == NULL || wh_rows == NULL) {
        return 0.0;
    }

    if (shift >= 0) {
        w_start = 0U;
        wh_start = (size_t)shift;
        if (wh_start >= wh_rows->count) {
            return 0.0;
        }
        length = w_rows->count;
        if (wh_rows->count - wh_start < length) {
            length = wh_rows->count - wh_start;
        }
    } else {
        w_start = (size_t)(-shift);
        wh_start = 0U;
        if (w_start >= w_rows->count) {
            return 0.0;
        }
        length = wh_rows->count;
        if (w_rows->count - w_start < length) {
            length = w_rows->count - w_start;
        }
    }

    if (length == 0U) {
        return 0.0;
    }

    for (i = 0; i < length; ++i) {
        const char *w_ts = w_rows->rows[w_start + i].timestamp;
        const char *wh_ts = wh_rows->rows[wh_start + i].timestamp;
        if (w_ts != NULL && wh_ts != NULL && strcmp(w_ts, wh_ts) == 0) {
            matched += 1U;
        }
    }

    return (100.0 * (double)matched) / (double)length;
}

int main(int argc, char **argv) {
    const int w_columns[FIELD_COUNT] = {1, 2, 3, 4, 5};
    const int wh_columns[FIELD_COUNT] = {1, 2, 3, 4, 7};
    const char *data_dir;
    const char *house_id;
    int min_shift;
    int max_shift;
    int legacy_mode = 0;
    double tolerance = 0.05;
    char w_path[1024];
    char wh_path[1024];
    RowBuffer w_rows = {0};
    RowBuffer wh_rows = {0};
    RowBuffer w_rows_processed = {0};
    RowBuffer wh_rows_processed = {0};
    const RowBuffer *w_eval = &w_rows;
    const RowBuffer *wh_eval = &wh_rows;
    int field_index;

    if (argc < 3 || argc > 7) {
        fprintf(
            stderr,
            "Usage: %s <data_dir> <house_id> [min_shift] [max_shift] [tolerance] [legacy_mode]\n",
            argv[0]
        );
        return 2;
    }

    data_dir = argv[1];
    house_id = argv[2];
    min_shift = (argc >= 4) ? atoi(argv[3]) : -30;
    max_shift = (argc >= 5) ? atoi(argv[4]) : 30;
    if (argc >= 6) {
        tolerance = strtod(argv[5], NULL);
    }
    if (argc >= 7) {
        legacy_mode = atoi(argv[6]) != 0;
    }

    if (min_shift > max_shift) {
        fprintf(stderr, "Invalid shift range: min_shift > max_shift\n");
        return 2;
    }

    snprintf(w_path, sizeof(w_path), "%s/H%s_W.csv", data_dir, house_id);
    snprintf(wh_path, sizeof(wh_path), "%s/H%s_Wh.csv", data_dir, house_id);

    if (!load_file_rows(w_path, w_columns, &w_rows)) {
        free_row_buffer(&w_rows);
        free_row_buffer(&wh_rows);
        return 1;
    }

    if (!load_file_rows(wh_path, wh_columns, &wh_rows)) {
        free_row_buffer(&w_rows);
        free_row_buffer(&wh_rows);
        return 1;
    }

    if (legacy_mode) {
        if (!preprocess_like_energy_plots(&w_rows, &w_rows_processed) ||
            !preprocess_like_energy_plots(&wh_rows, &wh_rows_processed)) {
            fprintf(stderr, "Failed while applying legacy preprocessing mode\n");
            free_row_buffer(&w_rows);
            free_row_buffer(&wh_rows);
            free_row_buffer(&w_rows_processed);
            free_row_buffer(&wh_rows_processed);
            return 1;
        }
        w_eval = &w_rows_processed;
        wh_eval = &wh_rows_processed;
    }

    printf("House H%s\n", house_id);
    printf("W rows:  %zu\n", w_eval->count);
    printf("Wh rows: %zu\n", wh_eval->count);
    printf("Shift sweep: %d..%d minutes, tolerance=%.4f\n\n", min_shift, max_shift, tolerance);
    printf("Legacy preprocessing mode: %s\n\n", legacy_mode ? "enabled" : "disabled");

    for (field_index = 0; field_index < FIELD_COUNT; ++field_index) {
        ShiftResult best = find_best_shift(
            w_eval,
            wh_eval,
            field_index,
            min_shift,
            max_shift,
            tolerance
        );

        if (!best.has_result) {
            printf("%s: no comparable rows in tested shift range\n", FIELD_NAMES[field_index]);
            continue;
        }

        printf(
            "%s: best_shift=%+d min, n=%zu, mae=%.6f, R2=%.6f, within_tol=%.2f%%, ts_match=%.2f%%\n",
            FIELD_NAMES[field_index],
            best.shift,
            best.compared,
            best.mae,
            best.r_squared,
            best.within_tolerance_pct,
            timestamp_match_pct_for_shift(w_eval, wh_eval, best.shift)
        );
    }

    printf("\nTimestamp match (shift -1): %.2f%%\n", timestamp_match_pct_for_shift(w_eval, wh_eval, -1));
    printf("Timestamp match (shift  0): %.2f%%\n", timestamp_match_pct_for_shift(w_eval, wh_eval, 0));

    free_row_buffer(&w_rows);
    free_row_buffer(&wh_rows);
    free_row_buffer(&w_rows_processed);
    free_row_buffer(&wh_rows_processed);
    return 0;
}
