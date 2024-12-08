
  Congratulations!  You now have a scanner calibration photo.
  Some important things you should keep in mind:

- This CD does contain the IT 8.7/2 reference file for the
  calibration target charge "R240103". Select the file "R240103.txt"
  as reference file when using the scanner profiling software.
  You can always download the latest reference files
  from http://www.targets.coloraid.de .

- Handle the targets carefully to avoid kink marks, scratches
  and fingerprints. Do not touch the surface.

- Remove the protection sleeve of the photo before scanning.
  Return the test target to its protective enclosure immediately
  after use.

- Always protect the calibration photo from strong light.
  Store prints in the dark at 24°C (75°F) or lower and at
  30- to 50-percent relative humidity. Avoid sudden
  temperature changes as this can cause moisture on the
  photo surface. Use a dry lint free cloth to dry/clean
  the surface. DO NOT USE ALCOHOL or other chemicals to
  clean the surface of the photo.

- The colors of the target will change with time. For critical work
  it is recommended to get a new target after 2028. 
  You can use the target much longer for non-critical work without
  risking major problems but you should expect small visible faults
  with such an age. Note that the production date printed
  on the target can be much older than the important measurement
  date noted under CREATED in the reference file. This allows colors
  to stabilize before measurement and ensures there is no 
  production fault causing increased aging.
  
  Please check out http://www.targets.coloraid.de
  for the latest information on your charge.

- You can use the calibration photo with basically any proper 
  ANSI IT 8.7/2 compatible software. Note that the reference file 
  content is identical for all operating systems and the file can 
  simply be transferred to other systems in order to get used.
  
- Some profiling programs expect reference files with the filename
  extension .it8 or .q60 . In this case you can simply rename
  the shipped .txt reference file. The IT8.7 standard
  defines the content of the file, but not the filename extension.
  As a result there are different filename extension used for the same
  file content.

  Some older profilers do expect reference files with the .tdf filename
  extension. The Kodak target description file (TDF) is a binary file.
  You can convert the shipped IT8 file to TDF using a tool from Kodaks
  FTP server: ftp://ftp.kodak.com/GASTDS/Q60DATA/TDF_FILES
  
- Users of the BasICColor profiling software having problems loading
  the reference file should copy the reference file into the
  reference files directory for Agfa targets of their BasICColor software.
  BasICColor may need an additional description file
  found in the Agfa directory to load the IT8 reference file.
  Ignore the different Agfa colors in column 20-22 of the target
  displayed in the BasICColor. The software will use the real
  colors provided in the reference file. If there are still
  problems, check if the used BasICColor version is compatible
  with the used operating system.
  
- Check out http://www.coloraid.de for links to various scanner
  profiling software and other CMS tools.
  
- The "Extras" directory does contain additional files for your
  target:

  R240103.it8 :   Identical to R240103.txt. Only the filename extension
                  was changed to .it8 as expected by some profilers.

  R240103W.txt :  Same as the normal reference file, but measurement
                  was done using a white instead of black backing.
                  The IT 8.7/2 standard requires the measurement of the
                  calibration target using a black backing.
                  Some (especially consumer) scanners do have a white
                  backing. Use this special reference file if your
                  scanner does ship with a white lid.

  R240103S.txt :  A smaller version of the normal reference file not
                  containing any statistical information on the target
                  production, resulting in a smaller file.
                  Use this reference file if your profiling
                  software does not accept the standard reference file.
                  Currently there seem to by only three profiling
                  programs not able to accept the standard reference
                  files. A fault in some versions of the
                  Heidelberg CPS ScanOpen software do cause a crash
                  when reading the large reference file.
                  Agfa's ColorTune users should also use this smaller
                  file in case the normal file is not accepted
                  by the profiler.
                  ColorQuartet (tested with 5.2.2 build 1) also
                  shows an error when loading the normal file.
                  If you still have problems loading the file
                  into ScanOpen, ColorQuartet or Colortune contact
                  wfaust@coloraid.de .

  R240103S.mrf :  This is basicly a gzip compressed version of 
                  the R240103S.txt file. The use of this file is
                  experimental and has not been tested nor is there
                  any support.  Use it on your own risk.
                  If you run into problems use the original target
                  you got with the profiling software.

  R240103.hist :  This file is for use with the X-Rite ColorShop X program 
                  and does contain the spectral data of the target. Using 
                  the spectral data ColorShop is able to calculate a large 
                  number of color operations, display the gamut hull and 
                  more. Note that the file format only supports spectral 
                  data from 400 to 700nm. A fully functional but time 
                  limited demo version of the ColorShop X software for 
                  MacOS and Windows can be found on the X-Rite website: 
                  http://www.x-rite.com   For more information on the 
                  spectral data please read below.
                  
  R240103ISO.txt: ISO 12641-1:2016 reference file. With the exception
                  of the first line identical with the R240103.txt file.

- The spectral data files provided in the "Extras" directory are currently 
  mainly for use by experienced users and developers. Normal users 
  intending to profile their scanners do not need these files.

  The spectral data can mainly be used to calculate color data for different 
  observers or color spaces. For instance, if you want to know the color
  under D65 instead of D50 light, you can use ColorShop to calculate
  the values.

- Measurement is based on wavelengths in 3nm intervals.  The 3nm data is
  interpolated to 10nm data according to ISO 13655:2009(E) Annex I.
  For batch average measured productions the spectral data 
  available is the mean data of several targets measured. XYZ tristimulus 
  and other color spaces calculated from the mean spectral data can differ 
  from the mean color space values listed in the IT 8.7 reference file. 
  Status T density informations found in the reference file are according 
  to ANSI CGATS.5-1993.

Wolf Faust - wfaust@coloraid.de
