# Installing Microsoft Visual C++ 14.0

A common issue that you might run into whilst installing dependencies, is the following:

![](../images/error.png)

If you look closely, it mentions that C++ is not installed which is required for some packages that are used in this book. 

![](../images/error1.png)

We can prevent this error either by installing the packages with conda using [this guide](README.md) or by installing Visual C++ separately. 

To do so, you first need to go to https://visualstudio.microsoft.com/visual-cpp-build-tools/ and download the build tools by clicking on "*Download Build Tools*":

![](../images/build_tools_1.png)

After doing so, you can start the application and press "*Continue*". It will download a number of small files needed for prepare the installer.

![](../images/build_tools_2.png)

When the installer is finished preparing you should see a screen that describes a number of things you can download and install. 

We are only interested in C++ so select "*Desktop development with C++*" to prepare a list of installs that we are interested in:

![](../images/build_tools_3.png)

After doing so, you will already see a number of installs checked. When you click "*Install*" it should install everything that is necessary for your environment. 

![](../images/build_tools_4.png)
