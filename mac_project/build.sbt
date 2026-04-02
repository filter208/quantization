scalaVersion := "2.13.12"

libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.6.1"
addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % "3.6.1" cross CrossVersion.full)

// 新增这一行：引入测试框架（% "test" 表示这个库只在测试时使用）
libraryDependencies += "edu.berkeley.cs" %% "chiseltest" % "0.6.2" % "test"