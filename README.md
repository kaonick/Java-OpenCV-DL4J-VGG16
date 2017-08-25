# Java-OpenCV-DL4J-VGG16
Real time object classification,Using JAVA,OpenCV,DL4J,VGG16-ImageNet

# Install OpenCV

# Download VGG16-ImageNet Model

# Project Maven dependency config 

        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-zoo</artifactId>
            <version>0.9.1</version>
        </dependency>

# Project Library config 
        C:/dev-soft/c_include/opencv3.2/build/java/opencv-320.jar

# Run config
        VM:
        -Djava.library.path=C:\dev-soft\c_include\opencv3.2\build\java\x64
        -Xms4G
        -Xmx8G
        -Dorg.bytedeco.javacpp.maxbytes=8G
        -Dorg.bytedeco.javacpp.maxPhysicalBytes=8G
