import Foundation

/// Promotes a fully copied bundle while retaining the previous installation
/// until the final rename succeeds. Staging and destination share a volume.
enum AppBundleInstaller {
    static func promote(staged: URL, to destination: URL, fileManager: FileManager = .default) throws {
        let backup = destination.deletingLastPathComponent()
            .appendingPathComponent(".VoiceFlow-backup-\(UUID().uuidString)")
        let hadPrevious = fileManager.fileExists(atPath: destination.path)
        if hadPrevious {
            try fileManager.moveItem(at: destination, to: backup)
        }
        do {
            try fileManager.moveItem(at: staged, to: destination)
        } catch {
            if hadPrevious {
                // A failed rollback must leave the backup intact for recovery.
                try fileManager.moveItem(at: backup, to: destination)
            }
            throw error
        }
        if hadPrevious { try? fileManager.removeItem(at: backup) }
    }
}
