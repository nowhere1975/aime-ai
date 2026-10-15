import AppKit
import Carbon
import RimeKit

/// Opt-in typing-process log for candidate-ranking research, off by default:
///   mkdir -p ~/Library/AIME/typing-log && touch ~/Library/AIME/typing-log/ENABLED
/// Delete ENABLED to stop. Writes one JSON object per line to
/// ~/Library/AIME/typing-log/YYYY-MM-DD.jsonl (0600), never leaves the Mac.
///
/// Records, per composition: up to 20 characters before the cursor when it starts, every
/// key with the pinyin and the candidate page it produced, and the commit. After a commit
/// only the *kind* of the next few keys is kept (to see ⌫ corrections), never characters
/// typed outside a composition. Nothing is written while Secure Input is on.
@MainActor
enum TypingLog {
    private static let dir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/AIME/typing-log")
    private static let flag = dir.appendingPathComponent("ENABLED")
    private static var buffer = ""
    private static var lastCheck = Date.distantPast
    private static var cachedEnabled = false
    /// Keys still to be classified after the last commit.
    private static var afterCommit = 0

    static var enabled: Bool {
        // A stat per keystroke is cheap, but once a second is enough.
        if Date().timeIntervalSince(lastCheck) > 1 {
            cachedEnabled = FileManager.default.fileExists(atPath: flag.path)
            lastCheck = Date()
        }
        return cachedEnabled && !IsSecureEventInputEnabled()
    }

    /// A key that may open a composition: letters without ⌘/⌃/⌥.
    static func mayStartComposition(_ event: NSEvent) -> Bool {
        guard event.modifierFlags.intersection([.command, .control, .option]).isEmpty,
              let c = event.charactersIgnoringModifiers?.unicodeScalars.first else { return false }
        return ("a"..."z").contains(Character(c))
    }

    /// Up to `limit` characters before the caret, read before any marked text exists.
    static func leftContext(_ client: (any IMKTextInput)?, limit: Int = 20) -> String? {
        guard let client else { return nil }
        let sel = client.selectedRange()
        guard sel.location != NSNotFound, sel.location > 0 else { return sel.location == 0 ? "" : nil }
        let start = max(0, sel.location - limit)
        return client.attributedSubstring(from: NSRange(location: start, length: sel.location - start))?.string
    }

    static func start(app: String?, context: String?, ms: Double) {
        write(["ev": "start", "app": app ?? "", "ctx": context as Any, "ctx_ms": (ms * 10).rounded() / 10])
    }

    /// Name of a key: the character for printable keys, a label for the rest.
    static func keyName(_ event: NSEvent) -> String {
        switch event.keyCode {
        case 51: return "bs"
        case 117: return "del"
        case 36, 76: return "ret"
        case 53: return "esc"
        case 48: return "tab"
        case 49: return "space"
        case 123: return "left"
        case 124: return "right"
        case 125: return "down"
        case 126: return "up"
        case 116: return "pgup"
        case 121: return "pgdn"
        default:
            let mods = event.modifierFlags.intersection([.command, .control, .option])
            if !mods.isEmpty { return "mod" }
            return event.characters ?? "?"
        }
    }

    static func key(_ event: NSEvent, wasComposing: Bool, handled: Bool, context: RimeContextSnapshot,
                    rankedIndex: Int? = nil) {
        let composing = context.isComposing
        if wasComposing || composing {
            var obj: [String: Any] = ["ev": "key", "k": keyName(event), "h": handled, "c0": wasComposing]
            if composing {
                obj["in"] = context.input
                obj["pre"] = context.composition.preedit
                obj["pg"] = context.pageNumber
                obj["hl"] = context.highlightedIndex
                obj["cands"] = context.candidates.map(\.text)
                if let rankedIndex { obj["rk"] = rankedIndex } // highlight moved by Ranker
            }
            write(obj)
        } else if afterCommit > 0 {
            // Outside a composition: only whether it was a correction key.
            afterCommit -= 1
            let name = keyName(event)
            write(["ev": "post", "k": ["bs", "del", "left", "right", "up", "down"].contains(name) ? name : "other"])
        }
    }

    static func commit(_ text: String) {
        afterCommit = 8
        write(["ev": "commit", "text": text])
        flush()
    }

    static func click(index: Int) { write(["ev": "click", "idx": index]) }
    static func pageClick(backward: Bool) { write(["ev": "pageclick", "back": backward]) }

    private static func write(_ object: [String: Any]) {
        var obj = object
        obj["t"] = Int(Date().timeIntervalSince1970 * 1000)
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
              let line = String(data: data, encoding: .utf8) else { return }
        buffer += line + "\n"
        if buffer.utf8.count > 16_384 { flush() }
    }

    static func flush() {
        guard !buffer.isEmpty else { return }
        let day = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate])
        let url = dir.appendingPathComponent("\(day).jsonl")
        let fm = FileManager.default
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        if !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
        }
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(buffer.utf8))
            try? handle.close()
        }
        buffer = ""
    }
}
