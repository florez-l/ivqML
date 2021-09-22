// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Linear__h__
#define __PUJ__Linear__h__

#include <PUJ/Traits.h>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
    class Linear
    {
    public:
      PUJ_TraitsMacro( Linear );

    public:
      Linear( const TRow& w, const TScalar& b );
      Linear( const TMatrix& X, const TCol& y );
      virtual ~Linear( ) = default;

      unsigned long GetDimensions( ) const;
      const TRow& GetWeights( ) const;
      const TScalar& GetBias( ) const;

      void SetWeights( const TRow& w );
      void SetBias( const TScalar& b );

      TScalar operator()( const TRow& x ) const;
      TCol operator()( const TMatrix& x ) const;

    protected:
      void _StreamIn( std::istream& i );
      void _StreamOut( std::ostream& o ) const;

    protected:
      TRow    m_Weights;
      TScalar m_Bias;

    public:
      /**
       */
      class Cost
      {
      public:
        Cost( const TMatrix& X, const TCol& y );
        virtual ~Cost( ) = default;

        TScalar operator()( const TRow& t, TRow* g = nullptr ) const;

      protected:
        TMatrix m_XtX;
        TRow    m_uX;
        TRow    m_Xy;
        TScalar m_uy;
        TScalar m_yty;
      };

    public:
      friend std::istream& operator>>( std::istream& i, Self& m )
        {
          m._StreamIn( i );
          return( i );
        }
      friend std::ostream& operator<<( std::ostream& o, const Self& m )
        {
          m._StreamOut( o );
          return( o );
        }
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Linear__h__

// eof - $RCSfile$
