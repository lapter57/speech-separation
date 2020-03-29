package com.autosub.subtitles;

import com.autosub.utils.SpeechRecognitionUtils;
import com.google.cloud.speech.v1.WordInfo;
import org.jetbrains.annotations.NotNull;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.TimeZone;
import java.util.stream.Collectors;

import static com.autosub.utils.SpeechRecognitionUtils.endTimeOf;
import static com.autosub.utils.SpeechRecognitionUtils.startTimeOf;
import static com.autosub.utils.SpeechRecognitionUtils.subtitleFrom;

public class SrtBlock {
    @NotNull
    private static final String TIME_DELIMITER = " --> ";
    @NotNull
    private final DateFormat formatter = new SimpleDateFormat("HH:mm:ss,SSS");
    private static final long SPLIT_TIME_BLOCKS_MS = 3000;

    private long id;
    private long startTimeInMs;
    private long endTimeInMs;
    @NotNull
    private List<String> subtitles;

    private SrtBlock(final long id,
                     final long startTimeInMs,
                     final long endTimeInMs) {
        this.id = id;
        this.startTimeInMs = startTimeInMs;
        this.endTimeInMs = endTimeInMs;
        formatter.setTimeZone(TimeZone.getTimeZone("GMT"));
        subtitles = new ArrayList<>();
    }

    private SrtBlock(final long id,
                     final long startTimeInMs,
                     final long endTimeInMs,
                     @NotNull final List<String> subtitles) {
        this(id, startTimeInMs, endTimeInMs);
        this.subtitles = subtitles;
    }

    public static List<SrtBlock> from(@NotNull final List<List<WordInfo>> speakers) {
        return from(SPLIT_TIME_BLOCKS_MS, speakers);
    }

    public static List<SrtBlock> from(final long splitTimeInMs,
                                      @NotNull final List<List<WordInfo>> speakers) {
        final var srtBlocks = new ArrayList<SrtBlock>();
        final long finalEndTime = speakers.stream()
                .flatMap(Collection::stream)
                .mapToLong(SpeechRecognitionUtils::endTimeOf)
                .max().orElse(0);
        long id = 1;
        while (splitTimeInMs * id <= (finalEndTime + splitTimeInMs) / splitTimeInMs * splitTimeInMs) {
            final var endTimeBlock = splitTimeInMs * id;
            final long startTimeBlock = endTimeBlock - splitTimeInMs;
            final var srtBlock = new SrtBlock(id, startTimeBlock, endTimeBlock);
            for (int i = 0; i < speakers.size(); i++) {
                final var words = speakers.get(i).stream()
                        .filter(srtBlock::isWordInBlock)
                        .collect(Collectors.toList());
                if (words.size() != 0) {
                    final var subtitle = subtitleFrom(words);
                    srtBlock.addSubtitle(subtitle);
                }
            }
            if (srtBlock.getSubtitles().size() != 0) {
                srtBlocks.add(srtBlock);
            }
            id++;
        }
        return srtBlocks;
    }

    private boolean isWordInBlock(@NotNull final WordInfo word) {
        final boolean exactlyIn = startTimeOf(word) > startTimeInMs &&
                endTimeOf(word) <= endTimeInMs;
        final boolean firstHalfInBlock = startTimeOf(word) < startTimeInMs &&
                startTimeInMs - startTimeOf(word) < endTimeOf(word) - startTimeInMs;
        final boolean secondHalfInBlock = endTimeOf(word) > endTimeInMs &&
                endTimeInMs - startTimeOf(word) >= endTimeOf(word) - endTimeInMs;
        return exactlyIn || firstHalfInBlock || secondHalfInBlock;
    }

    public long getId() {
        return id;
    }

    public long getStartTimeInMs() {
        return startTimeInMs;
    }

    public long getEndTimeInMs() {
        return endTimeInMs;
    }

    @NotNull
    public List<String> getSubtitles() {
        return subtitles;
    }

    private void addSubtitle(@NotNull final String subtitle) {
        this.subtitles.add(subtitle);
    }

    @Override
    public String toString() {
        final var str = new StringBuilder()
                .append(id)
                .append("\n")
                .append(formatter.format(new Date(startTimeInMs)))
                .append(TIME_DELIMITER)
                .append(formatter.format(new Date(endTimeInMs)))
                .append("\n");
        for (final var sub : subtitles) {
            str.append("- ").append(sub).append("\n");
        }
        return str.toString();
    }
}
