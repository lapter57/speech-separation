package com.autosub.services.speech.google;

import com.google.cloud.speech.v1.RecognitionAudio;
import com.google.cloud.speech.v1.RecognitionConfig;
import com.google.cloud.speech.v1.RecognitionConfig.AudioEncoding;
import com.google.cloud.speech.v1.SpeechContext;
import com.google.protobuf.ByteString;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public class GoogleS2TConfig {
    @NotNull private final RecognitionConfig config;

    private GoogleS2TConfig(@NotNull final Builder builder) {
        final var recConfig = RecognitionConfig.newBuilder()
                .setEncoding(builder.audioEncoding)
                .setSampleRateHertz(builder.sampleRateHertz.getHz())
                .setAudioChannelCount(builder.audioChannelCount)
                .setEnableSeparateRecognitionPerChannel(builder.enableSeparateRecognitionPerChannel)
                .setLanguageCode(builder.langCode.toString())
                .setMaxAlternatives(builder.maxAlternatives)
                .setProfanityFilter(builder.profanityFilter)
                .setEnableWordTimeOffsets(builder.enableWordTimeOffsets)
                .setModel(builder.model.toString())
                .setUseEnhanced(builder.useEnhanced);
        for (int i = 0; i < builder.speechContexts.size(); i++) {
            recConfig.setSpeechContexts(i, builder.speechContexts.get(i));
        }
        this.config = recConfig.build();
    }

    @NotNull
    public RecognitionConfig getConfig() {
        return config;
    }

    @NotNull
    public static Builder newBuilder() {
        return new Builder();
    }

    public static final class Builder {
        @NotNull private final LangCode langCode;
        @NotNull private final AudioEncoding audioEncoding;

        @NotNull private SampleRateHertz sampleRateHertz = SampleRateHertz.HZ_16000;
        @NotNull private List<SpeechContext> speechContexts = new ArrayList<>();
        @NotNull private Model model = Model.DEFAULT;
        private int audioChannelCount = 1;
        private int maxAlternatives = 1;
        private boolean enableSeparateRecognitionPerChannel = false;
        private boolean enableWordTimeOffsets = true;
        private boolean profanityFilter = false;
        private boolean useEnhanced = false;

        public Builder() {
            this(LangCode.EN_US);
        }

        public Builder(@NotNull final LangCode langCode) {
            this(langCode, AudioEncoding.LINEAR16);
        }

        public Builder(@NotNull final LangCode langCode,
                       @NotNull final AudioEncoding audioEncoding) {
            this.langCode = langCode;
            this.audioEncoding = audioEncoding;
        }

        public Builder setSampleRateHertz(@NotNull final SampleRateHertz sampleRateHertz) {
            this.sampleRateHertz = sampleRateHertz;
            return this;
        }

        public Builder setSpeechContexts(@NotNull final List<SpeechContext> speechContexts) {
            this.speechContexts = speechContexts;
            return this;
        }

        public Builder setModel(@NotNull final Model model) {
            this.model = model;
            return this;
        }

        public Builder setAudioChannelCount(final int audioChannelCount) {
            this.audioChannelCount = audioChannelCount;
            return this;
        }

        public Builder setMaxAlternatives(final int maxAlternatives) {
            this.maxAlternatives = maxAlternatives;
            return this;
        }

        public Builder setEnableSeparateRecognitionPerChannel(final boolean enableSeparateRecognitionPerChannel) {
            this.enableSeparateRecognitionPerChannel = enableSeparateRecognitionPerChannel;
            return this;
        }

        public Builder setEnableWordTimeOffsets(final boolean enableWordTimeOffsets) {
            this.enableWordTimeOffsets = enableWordTimeOffsets;
            return this;
        }

        public Builder setProfanityFilter(final boolean profanityFilter) {
            this.profanityFilter = profanityFilter;
            return this;
        }

        public Builder setUseEnhanced(final boolean useEnhanced) {
            this.useEnhanced = useEnhanced;
            return this;
        }

        @NotNull
        public GoogleS2TConfig build() {
            return new GoogleS2TConfig(this);
        }
    }

    public enum LangCode {
        RU_RU("ru-RU"),
        EN_US("en-US");

        @NotNull private final String code;

        LangCode(@NotNull final String code) {
            this.code = code;
        }

        @Override
        public String toString() {
            return code;
        }

        @NotNull
        public static LangCode fromString(@NotNull final String name) {
            for (final var value : LangCode.values()) {
                if (name.equalsIgnoreCase(value.code)) {
                    return value;
                }
            }
            throw new IllegalArgumentException("No lang code specified for this string");
        }
    }

    public enum Model {
        DEFAULT("default"),
        VIDEO("video"),
        PHONE_CALL("phone_call"),
        COMMAND_AND_SEARCH("command_and_search");

        @NotNull private final String model;

        Model(@NotNull final String model) {
            this.model = model;
        }

        @Override
        public String toString() {
            return model;
        }

        @NotNull
        public static Model fromString(@NotNull final String name) {
            for (final var value : Model.values()) {
                if (name.equalsIgnoreCase(value.model)) {
                    return value;
                }
            }
            throw new IllegalArgumentException("No model specified for this string");
        }
    }

    public enum SampleRateHertz {
        HZ_48000(48000),
        HZ_16000(16000),
        HZ_8000(8000);

        private final int hz;

        SampleRateHertz(final int hz) {
            this.hz = hz;
        }

        public int getHz() {
            return hz;
        }

        @Override
        public String toString() {
            return Integer.toString(hz);
        }

        @NotNull
        public static AudioEncoding fromString(@NotNull final String hz) {
            for (final var value : AudioEncoding.values()) {
                if (hz.equals(value.toString())) {
                    return value;
                }
            }
            throw new IllegalArgumentException("No sample rate specified for this string");
        }
    }
}
