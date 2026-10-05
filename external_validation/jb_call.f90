program jb_call
  implicit none
  integer :: ios
  real(8) :: amjd, sun(2), sat(3), f10, f10b, s10, s10b
  real(8) :: xm10, xm10b, y10, y10b, dstdtc, temp(2), rho
  do
    read(*,*,iostat=ios) amjd, sun(1), sun(2), sat(1), sat(2), sat(3), &
      f10, f10b, s10, s10b, xm10, xm10b, y10, y10b, dstdtc
    if (ios /= 0) exit
    call JB2008(amjd, sun, sat, f10, f10b, s10, s10b, xm10, xm10b, &
      y10, y10b, dstdtc, temp, rho)
    write(*,'(ES24.16)') rho
  end do
end program jb_call
