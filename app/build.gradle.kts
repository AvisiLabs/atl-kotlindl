plugins {
    id("org.jetbrains.kotlin.jvm") version "1.6.10"

    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(platform("org.jetbrains.kotlin:kotlin-bom"))
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
    implementation("ch.qos.logback:logback-classic:1.2.10")

    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.3.0")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-onnx:0.3.0")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-visualization:0.3.0")
}
