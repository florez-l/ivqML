// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include "MineSweeperPlayerBase.h"

// -------------------------------------------------------------------------
MineSweeperPlayerBase::
MineSweeperPlayerBase( )
{
}

// -------------------------------------------------------------------------
void MineSweeperPlayerBase::
configure( unsigned int w, unsigned int h, unsigned long n )
{
  this->m_Width = w;
  this->m_Height = h;
  this->m_NumberOfMines = n;
}

// -------------------------------------------------------------------------
void MineSweeperPlayerBase::
report( const unsigned char& c )
{
}

// eof - $RCSfile$
