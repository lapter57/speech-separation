import net.ltgt.gradle.errorprone.errorprone

plugins {
    java
    application
    id("net.ltgt.errorprone") version "0.8.1"
}

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

repositories {
    jcenter()
}

dependencies {
    // Checks
    errorprone("com.google.errorprone:error_prone_core:2.3.3")

    // Logging
    compile("org.slf4j:slf4j-api:1.7.26")
    compile("ch.qos.logback:logback-classic:1.2.3")

    // Annotations for better code documentation
    compile("com.intellij:annotations:12.0")

    // Guava primitives
    compile("com.google.guava:guava:27.0.1-jre")

    // Lombok
    compile("org.projectlombok:lombok:1.18.8")
    annotationProcessor("org.projectlombok:lombok:1.18.12")

    // Google Cloud
    compile ("com.google.cloud:google-cloud-speech:1.22.5")

    // Spring boot
    compile("org.springframework.boot:spring-boot-starter-web:2.2.6.RELEASE")
    compile("org.springframework.boot:spring-boot-devtools:2.2.6.RELEASE")
    testCompile("org.springframework.boot:spring-boot-starter-test:2.2.6.RELEASE")

    // JUnit Jupiter test framework
    testCompile("org.junit.jupiter:junit-jupiter-api:5.4.0")
    testRuntime("org.junit.jupiter:junit-jupiter-engine:5.4.0")
}

val run by tasks.getting(JavaExec::class) {
    standardInput = System.`in`
}

tasks {
    test {
        useJUnitPlatform()
    }
}

application {
    // Define the main class for the application
    mainClassName = "com.autosub.Application"
}

// Error prone options
tasks.named<JavaCompile>("compileTestJava") {
    options.errorprone.isEnabled.set(false)
}
