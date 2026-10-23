import Carbon
import Darwin
import Foundation

/// Opt-in context-aware candidate ranking (research prototype), off by default:
///   touch ~/Library/AIME/ranker/ENABLED   and run research/ranker/server.py
/// Delete ENABLED to stop. For each new input on the first page, AIME asks a local service
/// (Unix socket in the same folder, owner-only) which candidate fits the text before the
/// caret best, then moves RIME's highlight there: the order and the number keys stay as
/// RIME made them, Space takes the highlighted one. No answer within `timeout` (or no
/// service) leaves RIME's own choice. Nothing is sent while Secure Input is on.
@MainActor
enum Ranker {
    private static let dir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/AIME/ranker")
    private static let flag = dir.appendingPathComponent("ENABLED")
    private static let socketPath = dir.appendingPathComponent("ranker.sock").path
    static let timeout: TimeInterval = 0.05
    private static var fd: Int32 = -1
    private static var retryAt = Date.distantPast
    private static var lastCheck = Date.distantPast
    private static var cachedEnabled = false

    static var enabled: Bool {
        if Date().timeIntervalSince(lastCheck) > 1 {
            cachedEnabled = FileManager.default.fileExists(atPath: flag.path)
            lastCheck = Date()
            if !cachedEnabled, fd >= 0 { disconnect(retryAfter: 0) }
        }
        return cachedEnabled && !IsSecureEventInputEnabled()
    }

    /// Index on the current page to highlight, or nil to keep RIME's choice.
    static func rank(context: String, previous: String, candidates: [String]) -> Int? {
        guard send(["op": "rank", "ctx": context, "prev": previous, "cands": candidates]) else { return nil }
        guard let line = readLine(deadline: Date().addingTimeInterval(timeout)),
              let reply = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
              let index = reply["i"] as? Int else { return nil }
        return index
    }

    /// Tells the service what was committed (personal history, next context). No reply.
    static func commit(previous: String, text: String) {
        _ = send(["op": "commit", "prev": previous, "text": text])
    }

    // MARK: - Socket

    private static func connectIfNeeded() -> Bool {
        if fd >= 0 { return true }
        guard Date() >= retryAt else { return false }
        let s = socket(AF_UNIX, SOCK_STREAM, 0)
        guard s >= 0 else { retryAt = Date().addingTimeInterval(5); return false }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let path = Array(socketPath.utf8CString)
        guard path.count <= MemoryLayout.size(ofValue: addr.sun_path) else { close(s); return false }
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            path.withUnsafeBytes { raw.copyMemory(from: $0) }
        }
        var on: Int32 = 1
        setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
        var tv = timeval(tv_sec: 0, tv_usec: Int32(timeout * 1_000_000))
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        let ok = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(s, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) == 0
            }
        }
        guard ok else {
            close(s)
            retryAt = Date().addingTimeInterval(5) // service not running: don't try on every key
            return false
        }
        fd = s
        return true
    }

    /// A late reply must never be read as the answer to the next request, so a timeout
    /// drops the connection; the next key reconnects at once.
    private static func disconnect(retryAfter seconds: TimeInterval) {
        if fd >= 0 { close(fd) }
        fd = -1
        retryAt = Date().addingTimeInterval(seconds)
    }

    private static func send(_ object: [String: Any]) -> Bool {
        guard connectIfNeeded(),
              var data = try? JSONSerialization.data(withJSONObject: object) else { return false }
        data.append(0x0A)
        let sent = data.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
        if sent != data.count { disconnect(retryAfter: 0); return false }
        return true
    }

    private static func readLine(deadline: Date) -> Data? {
        var buffer = Data()
        var chunk = [UInt8](repeating: 0, count: 512)
        while Date() < deadline {
            let n = read(fd, &chunk, chunk.count)
            if n <= 0 { break }
            buffer.append(contentsOf: chunk[0..<n])
            if let newline = buffer.firstIndex(of: 0x0A) { return buffer.prefix(upTo: newline) }
        }
        disconnect(retryAfter: 0)
        return nil
    }
}
