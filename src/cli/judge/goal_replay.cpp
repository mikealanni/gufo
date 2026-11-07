#include "src/cli/judge/goal_replay.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/jev_goal.hpp"

namespace gufo::cli {
namespace {

namespace flash = models::qwen38_flash_next;
using Role = flash::JevGoalMessage::Role;

struct CallText {
  std::string id, name, command, path;
};

struct Event {
  Role role{Role::kUser};
  std::string text;
  std::vector<CallText> calls;
  std::string id, name;
};

struct Session {
  std::string file;
  std::string model;
  std::vector<Event> events;
  std::vector<flash::JevGoalMessage> messages;  // views into `events`
};

enum class Label { kStrong, kLoose, kOther, kEof };

constexpr std::array<std::string_view, 4> kLabelNames{
    {"strong", "loose", "other", "eof"}};

std::string Fold(std::string_view text) {
  std::string out(text);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

bool StartsWith(const std::string& folded, std::string_view prefix) {
  return folded.compare(0, prefix.size(), prefix) == 0;
}

// The user's next message tells whether the previous turn really was done.
Label LabelNext(std::string_view text, std::string_view goal = {}) {
  const std::string folded = Fold(text);
  constexpr std::size_t kResentChars = 40;
  if (goal.size() >= kResentChars && text.size() >= kResentChars &&
      Fold(text.substr(0, kResentChars)) ==
          Fold(goal.substr(0, kResentChars))) {
    return Label::kStrong;
  }
  constexpr std::array<std::string_view, 13> kStrongStart{{
      "continue",
      "keep going",
      "go on",
      "proceed",
      "carry on",
      "finish",
      "resume",
      "do it",
      "go ahead",
      "try again",
      "yes",
      "ok continue",
      "complete",
  }};
  constexpr std::array<std::string_view, 16> kStrongAnywhere{{
      "you didn't",
      "you didnt",
      "not finished",
      "unfinished",
      "incomplete",
      "you stopped",
      "keep going",
      "still not",
      "not done",
      "you missed",
      "you forgot",
      "why did you stop",
      "didn't finish",
      "without stopping",
      "to the end",
      "don't stop",
  }};
  for (const auto prefix : kStrongStart) {
    if (StartsWith(folded, prefix))
      return Label::kStrong;
  }
  if (StartsWith(folded, "go") &&
      (folded.size() == 2 ||
       !std::isalnum(static_cast<unsigned char>(folded[2]))))
    return Label::kStrong;
  for (const auto needle : kStrongAnywhere) {
    if (folded.find(needle) != std::string::npos)
      return Label::kStrong;
  }
  constexpr std::array<std::string_view, 7> kLoose{{
      "also",
      "and ",
      "now ",
      "next",
      "what about",
      "then ",
      "one more",
  }};
  for (const auto prefix : kLoose) {
    if (StartsWith(folded, prefix))
      return Label::kLoose;
  }
  return Label::kOther;
}

bool ContinueLike(std::string_view text) {
  constexpr std::size_t kShortChars = 60;
  return LabelNext(text) == Label::kStrong && text.size() < kShortChars;
}

bool LoadSession(const std::filesystem::path& path, Session* session) {
  std::ifstream in(path);
  if (!in)
    return false;
  session->file = path.filename().string();
  std::string line;
  bool first = true;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    json::Value value;
    try {
      value = json::parse(line);
    } catch (const std::exception&) {
      continue;
    }
    if (first) {
      first = false;
      if (value.find("session")) {
        session->model = value.member_str("model");
        continue;
      }
    }
    Event event;
    const std::string role = value.member_str("role");
    event.role = role == "assistant" ? Role::kAssistant
                 : role == "tool"    ? Role::kTool
                                     : Role::kUser;
    event.text = value.member_str("text");
    event.id = value.member_str("id");
    event.name = value.member_str("name");
    if (const auto* calls = value.find("calls"); calls && calls->is_array()) {
      for (const auto& call : calls->items()) {
        event.calls.push_back({call.member_str("id"), call.member_str("name"),
                               call.member_str("command"),
                               call.member_str("path")});
      }
    }
    session->events.push_back(std::move(event));
  }
  session->messages.reserve(session->events.size());
  for (const auto& event : session->events) {
    flash::JevGoalMessage message;
    message.role = event.role;
    message.text = event.text;
    message.tool_call_id = event.id;
    message.tool_name = event.name;
    for (const auto& call : event.calls) {
      message.calls.push_back({call.id, call.name, call.command, call.path});
    }
    session->messages.push_back(std::move(message));
  }
  return !session->events.empty();
}

struct Point {
  const Session* session;
  std::size_t end;  // index of the final assistant event
  flash::JevGoalSpec spec;
  Label label;
  bool flash_next;
};

std::string OneLine(std::string_view text, std::size_t limit, bool tail) {
  std::string out(tail && text.size() > limit ? text.substr(text.size() - limit)
                                              : text.substr(0, limit));
  for (auto& c : out)
    c = c == '\n' ? ' ' : c;
  return out;
}

}  // namespace

int RunGoalReplay(const std::string& dir, bool verbose, bool bare) {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() == ".jsonl")
      files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  std::vector<Session> sessions(files.size());
  std::size_t loaded = 0;
  for (std::size_t i = 0; i < files.size(); ++i) {
    if (LoadSession(files[i], &sessions[loaded]))
      ++loaded;
  }
  sessions.resize(loaded);

  std::vector<Point> points;
  std::size_t episodes = 0;
  for (const auto& session : sessions) {
    const auto& events = session.events;
    flash::JevGoalSpec goal;
    for (std::size_t u = 0; u < events.size(); ++u) {
      if (events[u].role != Role::kUser)
        continue;
      ++episodes;
      if (!goal.active || !ContinueLike(events[u].text)) {
        goal.active = true;
        goal.text = events[u].text;
        goal.index = u;
        goal.task_index = u;
      }
      std::size_t next_user = u + 1;
      while (next_user < events.size() && events[next_user].role != Role::kUser)
        ++next_user;
      if (next_user == u + 1)
        continue;
      const std::size_t end = next_user - 1;
      if (events[end].role != Role::kAssistant || !events[end].calls.empty())
        continue;
      Point point;
      point.session = &session;
      point.end = end;
      point.spec = goal;
      point.label = next_user < events.size()
                        ? LabelNext(events[next_user].text, goal.text)
                        : Label::kEof;
      point.flash_next = session.model.find("Flash Next") != std::string::npos;
      points.push_back(std::move(point));
    }
  }

  struct Tally {
    std::array<int, 4> by_label{};
  };
  std::array<Tally, 2> verdicts;  // [continue-or-nudge, done]
  std::array<Tally, 2> flash_verdicts;
  std::map<std::string, Tally> rules;
  std::vector<double> micros;
  micros.reserve(points.size());
  for (const auto& point : points) {
    const auto& session = *point.session;
    const auto t0 = std::chrono::steady_clock::now();
    const auto decision =
        flash::EvaluateJevGoal(point.spec,
                               std::span<const flash::JevGoalMessage>(
                                   session.messages.data(), point.end),
                               session.events[point.end].text, 0);
    micros.push_back(std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
    if (verbose && micros.back() > 2000.0) {
      std::printf("SLOW %.0f us history=%zu goal_chars=%zu final_chars=%zu\n",
                  micros.back(), point.end - point.spec.index,
                  point.spec.text.size(),
                  session.events[point.end].text.size());
    }
    const bool fired = decision.action == flash::JevGoalAction::kContinue ||
                       decision.action == flash::JevGoalAction::kNudge;
    const auto label = static_cast<std::size_t>(point.label);
    ++verdicts[fired ? 0 : 1].by_label[label];
    if (point.flash_next)
      ++flash_verdicts[fired ? 0 : 1].by_label[label];
    if (fired)
      ++rules[decision.rule].by_label[label];
    if (verbose && fired && point.label == Label::kOther) {
      const auto& events = session.events;
      std::size_t next = point.end + 1;
      std::printf(
          "FALSE-CONTINUE %s [%s] msg: %s | goal: %s | final: ...%s | next: "
          "%s\n",
          session.file.c_str(), decision.rule.c_str(),
          OneLine(decision.message, 110, false).c_str(),
          OneLine(point.spec.text, 110, false).c_str(),
          OneLine(events[point.end].text, 80, true).c_str(),
          next < events.size() ? OneLine(events[next].text, 60, false).c_str()
                               : "");
    }
  }

  const auto print = [](const char* name, const std::array<Tally, 2>& v) {
    const auto& c = v[0].by_label;
    const auto& d = v[1].by_label;
    const double strict_den = c[0] + c[1] + c[2];
    std::printf("%s decisions=%d\n", name,
                c[0] + c[1] + c[2] + c[3] + d[0] + d[1] + d[2] + d[3]);
    std::printf("  continue: strong=%d loose=%d other=%d eof=%d\n", c[0], c[1],
                c[2], c[3]);
    std::printf("  done:     strong=%d loose=%d other=%d eof=%d\n", d[0], d[1],
                d[2], d[3]);
    std::printf(
        "  precision_strict=%.3f precision_loose=%.3f recall_strict=%.3f "
        "recall_loose=%.3f false_continues=%d\n",
        strict_den > 0 ? c[0] / strict_den : 0.0,
        strict_den > 0 ? (c[0] + c[1]) / strict_den : 0.0,
        (c[0] + d[0]) > 0 ? static_cast<double>(c[0]) / (c[0] + d[0]) : 0.0,
        (c[0] + c[1] + d[0] + d[1]) > 0
            ? static_cast<double>(c[0] + c[1]) / (c[0] + c[1] + d[0] + d[1])
            : 0.0,
        c[2]);
  };
  std::printf("sessions=%zu user_episodes=%zu decision_points=%zu\n",
              sessions.size(), episodes, points.size());
  print("ALL", verdicts);
  print("FLASH-NEXT", flash_verdicts);

  if (bare) {
    // Bare phrase typed as the latest user message at every decision point;
    // the goal must come from earlier turns.
    static constexpr std::string_view kBare = "jev finish it";
    std::array<Tally, 2> bare_verdicts;
    std::size_t no_items = 0;
    std::size_t from_history = 0;
    std::size_t fired_fallback = 0;
    std::vector<double> bare_micros;
    std::map<const Session*, std::vector<flash::JevGoalMessage>> copies;
    for (const auto& point : points) {
      const auto& session = *point.session;
      auto& messages = copies[&session];
      if (messages.empty())
        messages = session.messages;
      std::size_t last_user = point.end;
      while (last_user > 0 && session.events[last_user].role != Role::kUser)
        --last_user;
      if (session.events[last_user].role != Role::kUser)
        continue;
      const std::string_view saved = messages[last_user].text;
      messages[last_user].text = kBare;
      const auto t0 = std::chrono::steady_clock::now();
      const auto spec = flash::DetectJevGoal(
          std::span<const flash::JevGoalMessage>(messages.data(), point.end));
      const auto decision = flash::EvaluateJevGoal(
          spec,
          std::span<const flash::JevGoalMessage>(messages.data(), point.end),
          session.events[point.end].text, 0);
      bare_micros.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t0)
                                .count());
      messages[last_user].text = saved;
      no_items += spec.items == 0;
      from_history += spec.from_history;
      const bool fired = decision.action == flash::JevGoalAction::kContinue ||
                         decision.action == flash::JevGoalAction::kNudge;
      fired_fallback += fired && decision.reason.find("no-items-fallback") !=
                                     std::string::npos;
      ++bare_verdicts[fired ? 0 : 1]
            .by_label[static_cast<std::size_t>(point.label)];
    }
    std::printf(
        "BARE-PHRASE points=%zu goal_from_history=%zu no_items=%zu "
        "fallback_fires=%zu\n",
        bare_micros.size(), from_history, no_items, fired_fallback);
    print("BARE-PHRASE", bare_verdicts);
    std::sort(bare_micros.begin(), bare_micros.end());
    if (!bare_micros.empty()) {
      double sum = 0.0;
      for (const double us : bare_micros)
        sum += us;
      std::printf(
          "bare timing detect+evaluate mean=%.1f us p50=%.1f us "
          "p99=%.1f us max=%.1f us\n",
          sum / bare_micros.size(), bare_micros[bare_micros.size() / 2],
          bare_micros[static_cast<std::size_t>(bare_micros.size() * 0.99)],
          bare_micros.back());
    }
  }
  for (const auto& [rule, tally] : rules) {
    const auto& c = tally.by_label;
    std::printf("rule %-16s fires=%d strong=%d loose=%d other=%d eof=%d\n",
                rule.c_str(), c[0] + c[1] + c[2] + c[3], c[0], c[1], c[2],
                c[3]);
  }

  // Timing: single-decision latency over >= 10k evaluations.
  std::vector<double> bench;
  bench.reserve(10000);
  while (!points.empty() && bench.size() < 10000) {
    for (const auto& point : points) {
      const auto& session = *point.session;
      const auto t0 = std::chrono::steady_clock::now();
      const auto decision =
          flash::EvaluateJevGoal(point.spec,
                                 std::span<const flash::JevGoalMessage>(
                                     session.messages.data(), point.end),
                                 session.events[point.end].text, 0);
      bench.push_back(std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - t0)
                          .count());
      if (decision.items > 1000000)
        std::printf("unreachable\n");
      if (bench.size() >= 10000)
        break;
    }
  }
  std::sort(bench.begin(), bench.end());
  if (!bench.empty()) {
    double sum = 0.0;
    for (const double us : bench)
      sum += us;
    std::printf(
        "timing evals=%zu mean=%.1f us p50=%.1f us p99=%.1f us max=%.1f us\n",
        bench.size(), sum / bench.size(), bench[bench.size() / 2],
        bench[static_cast<std::size_t>(bench.size() * 0.99)], bench.back());
  }

  // Trigger strip cost on a 1 KB user message.
  std::string text(900, 'x');
  text += " fix the parser in a.cpp, Jev, make sure it's done. then stop";
  const auto t0 = std::chrono::steady_clock::now();
  std::size_t sink = 0;
  for (int i = 0; i < 10000; ++i)
    sink += flash::StripJevGoalTrigger(text).size();
  const double strip_us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0)
                              .count() /
                          10000.0;
  std::printf("strip 1KB message: %.2f us (sink %zu)\n", strip_us, sink);
  return 0;
}

}  // namespace gufo::cli
