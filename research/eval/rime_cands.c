// Line-oriented driver around AIME's librime.
// stdin : one pinyin input per line (no separators), e.g. "youxiang"
// stdout: candidates for that input, tab-separated, one line per input (up to MAX_CANDS)
// Composition is cleared after each query, nothing is committed, so the user dictionary
// is not trained by the eval.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rime_api.h"  // from CRime.xcframework Headers (see setup.sh)

#define MAX_CANDS 20

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <shared_dir> <user_dir> <schema_id>\n", argv[0]);
    return 2;
  }
  RimeApi *api = rime_get_api();
  RIME_STRUCT(RimeTraits, traits);
  traits.shared_data_dir = argv[1];
  traits.user_data_dir = argv[2];
  traits.distribution_name = "AIME-eval";
  traits.distribution_code_name = "AIME";
  traits.distribution_version = "0.0.1";
  traits.app_name = "rime.aime-eval";
  traits.min_log_level = 2;
  api->setup(&traits);
  api->initialize(&traits);
  if (api->start_maintenance(False)) api->join_maintenance_thread();

  RimeSessionId s = api->create_session();
  if (!s || !api->select_schema(s, argv[3])) {
    fprintf(stderr, "cannot select schema %s\n", argv[3]);
    return 1;
  }
  api->set_option(s, "ascii_mode", False);
  fprintf(stderr, "ready\n");

  char line[1024];
  while (fgets(line, sizeof line, stdin)) {
    line[strcspn(line, "\r\n")] = 0;
    api->clear_composition(s);
    if (*line) api->set_input(s, line);
    RimeCandidateListIterator it = {0};
    int n = 0;
    if (api->candidate_list_begin(s, &it)) {
      while (n < MAX_CANDS && api->candidate_list_next(&it)) {
        if (n++) fputc('\t', stdout);
        fputs(it.candidate.text, stdout);
      }
      api->candidate_list_end(&it);
    }
    fputc('\n', stdout);
    fflush(stdout);
  }
  api->clear_composition(s);
  api->destroy_session(s);
  api->finalize();
  return 0;
}
