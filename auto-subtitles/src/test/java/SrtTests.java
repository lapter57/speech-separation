import com.autosub.services.speech.google.GoogleS2T;
import com.autosub.services.speech.google.GoogleS2TConfig;
import com.autosub.subtitles.SrtBlock;
import com.google.cloud.speech.v1.WordInfo;
import com.google.protobuf.ByteString;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.List;
import java.util.Objects;
import java.util.TimeZone;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardOpenOption.APPEND;
import static java.nio.file.StandardOpenOption.CREATE;

public class SrtTests {
    private static final DateFormat formatter = new SimpleDateFormat("HH:mm:ss,SSS");
    static {
        formatter.setTimeZone(TimeZone.getTimeZone("GMT"));
    }
    @Test
    public void merge2Speakers() throws Exception {
        final var executor = Executors.newFixedThreadPool(2);
        final var future1 = executor.submit(() -> recognize("src/test/resources/1.wav"));
        final var future2 = executor.submit(() -> recognize("src/test/resources/2.wav"));
        final var srtBlocks = SrtBlock.from(List.of(future1.get(), future2.get()));
        writeAlternativeToFile(srtBlocks, "src/test/resources/answer.srt");
        executor.shutdown();
    }

    private static List<WordInfo> recognize(@NotNull final String file) throws IOException {
        final var path = Paths.get(file);
        final var data = Files.readAllBytes(path);
        final var audioBytes = ByteString.copyFrom(data);
        final var config = GoogleS2TConfig.newBuilder().setModel(GoogleS2TConfig.Model.VIDEO).build();
        final var recognizer = GoogleS2T.of(config);
        final var answer = recognizer.recognize(audioBytes);
        return answer.stream()
                .map(srr -> srr.getAlternatives(0))
                .flatMap(a -> a.getWordsList().stream())
                .collect(Collectors.toList());
    }

    private static void writeAlternativeToFile(@NotNull final List<SrtBlock> blocks,
                                               @NotNull final String dest) {
        try(final var writer = new PrintWriter(Files.newBufferedWriter(Paths.get(dest), UTF_8, CREATE, APPEND))) {
            blocks.forEach(b -> writer.println(b.toString()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
